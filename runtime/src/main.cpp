// Boot: map and link the recompiled modules (what rtld does), create the main thread and enter
// nn::init::Start(main thread handle, main module base, init, fini) like rtld's _start.
// The platform's entry point (platform_win.cpp / platform_android.cpp) calls nx::boot().
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#include <nx/game.h>
#include <nx/runtime.h>

#include "kernel.h"
#include "platform.h"

#if NX_NCE
#include "game_image.h"
#include "nce/nce.h"
#include "nce/nso.h"
#else
extern const nx::ModuleInfo nx_module_main;
extern const nx::ModuleInfo nx_module_sdk;
#endif

namespace nx {

// The game's two modules: the recompiled ones, or (native execution) the NSOs of the player's exefs.
static const ModuleInfo* g_main = nullptr;
static const ModuleInfo* g_sdk = nullptr;

uint64_t mrs(CPU& c, uint32_t sysreg) {
    switch (sysreg) {
        case 0xDF02:  // CNTVCT_EL0
        case 0xDF01:  // CNTPCT_EL0
            return system_tick();
        case 0xDF00:  // CNTFRQ_EL0
            return 19200000;
        case 0xD807:  // DCZID_EL0: DC ZVA allowed, 64-byte blocks
            return 4;
        case 0xC000:  // MIDR_EL1 (Cortex-A57)
            return 0x411FD071;
        case 0xD801:  // CTR_EL0 (Cortex-A57: 64-byte lines, PIPT instruction cache)
            return 0x8444C004;
        default:
            log("mrs %#x at lr %s", sysreg, describe(c.x[30]).c_str());
            return 0;
    }
}

void msr(CPU& c, uint32_t sysreg, uint64_t value) {
    log("msr %#x = %llx at lr %s", sysreg, (unsigned long long)value, describe(c.x[30]).c_str());
}

// ---- module initializers --------------------------------------------------------------------------------------
// Each module's DT_INIT (at +0x30 in both main and sdk, see .dynamic) registers the module with the libc
// (__nnmusl_init_dso: TLS, unwind info) and runs its init_array, unless the registration reports 1: then the
// constructors are left to the loader and we run them. Registration only works once the program has given nn::os
// a TLS allocator (SetMemoryAllocatorForThreadLocal, in nninitStartup), so rtld's initializer callback does nothing
// and the modules are initialized right before nnMain, SDK first. Static TLS itself comes from the kernel
// (kernel.cpp: a block per thread at TLS region +0x1F0, where nn::os loads TPIDR_EL0 from).
constexpr uint64_t kDtInit = 0x30;

static void call_preserving(CPU& c, uint64_t target) {
    uint64_t saved[13];
    for (int r = 19; r <= 31; r++) saved[r - 19] = c.x[r];
    call(c, target);
    for (int r = 19; r <= 31; r++) c.x[r] = saved[r - 19];
}

// DT_INIT loads the word where it keeps __nnmusl_init_dso's result with `adrp xN, page` ... `ldr w8, [xN, #off]`
// in its first instructions (SDK 9: adjacent; SDK 16: another adrp in between). Follow the adrp targets until
// that load.
static uint64_t init_status_addr(const ModuleInfo& m, uint64_t start, bool must = true) {
    uint64_t page[32] = {};
    for (uint64_t pc = start; pc < start + 0x40; pc += 4) {
        uint32_t w = ld<uint32_t>(pc);
        if ((w & 0x9F000000) == 0x90000000) {  // adrp
            int64_t imm = int64_t(((w >> 5) & 0x7FFFF) << 2 | ((w >> 29) & 3)) << 43 >> 31;  // sign-extended << 12
            page[w & 31] = (pc & ~0xFFFull) + uint64_t(imm);
        } else if ((w & 0xFFC0001F) == 0xB9400008 && page[(w >> 5) & 31]) {  // ldr w8, [xN, #imm]
            return page[(w >> 5) & 31] + ((w >> 10) & 0xFFF) * 4;
        }
    }
    if (must) fatal("%s: DT_INIT does not load its init status as expected", m.name);
    return 0;
}

// DT_INIT hands __nnmusl_init_dso the module's layout: consecutive GOT slots it loads with adrp + ldr. Slots 13
// and 14 are the start and end of the module's .tdata (checked on SDK 16: Wonder's main 0x359cc18..0x359cc28,
// sdk empty). Returns false when the pattern is not there (SDK 9 modules have no .tdata to worry about).
static bool tdata_range(const ModuleInfo& m, uint64_t& start, uint64_t& end) {
    uint64_t page[32] = {}, lo = ~0ull, hi = 0;
    for (uint64_t pc = m.base + kDtInit; pc < m.base + kDtInit + 0x100; pc += 4) {
        uint32_t w = ld<uint32_t>(pc);
        if ((w & 0x9F000000) == 0x90000000) {  // adrp
            int64_t imm = int64_t(((w >> 5) & 0x7FFFF) << 2 | ((w >> 29) & 3)) << 43 >> 31;
            page[w & 31] = (pc & ~0xFFFull) + uint64_t(imm);
        } else if ((w & 0xFFC00000) == 0xF9400000 && page[(w >> 5) & 31]) {  // ldr xT, [xN, #imm]
            uint64_t slot = page[(w >> 5) & 31] + ((w >> 10) & 0xFFF) * 8;
            lo = std::min(lo, slot), hi = std::max(hi, slot);
        }
    }
    if (hi < lo || (hi - lo) / 8 + 1 < 15) return false;
    start = ld<uint64_t>(lo + 13 * 8), end = ld<uint64_t>(lo + 14 * 8);
    return start <= end && end - start <= 0x1000 && start >= m.base && end <= m.base + 0x10000000;
}

static uint64_t module_symbol(const ModuleInfo& m, const char* name);

// Where the module's DT_INIT code starts. SDK 9 and 16 modules have it at +0x30; an SDK 2017 module has its `_init`
// at +0x24 (the log's symbol list: _init at base+24), so +0x30 lands in the middle of its prologue and the call goes
// astray (pc 0). The module's own `_init` symbol wins, +0x30 is the fallback.
static uint64_t dt_init_addr(const ModuleInfo& m, bool* from_symbol) {
    const uint64_t sym = module_symbol(m, "_init");
    if (from_symbol) *from_symbol = sym != 0;
    return sym ? sym : m.base + kDtInit;
}

// A DT_INIT word is usable when it is not an all-zero or all-one word: 0x0000xxxx is the permanently undefined UDF
// encoding and 0xFFFFFFFF an erased word, both of which fault when executed.
static bool dt_init_usable(uint32_t word) { return (word >> 16) != 0 && word != 0xFFFFFFFFu; }

// True when the init_array entry's own code calls (or tail-jumps to) `target` within its first instructions: a static
// initializer that constructs one global object. Used to leave out the entry that builds the OS resource manager, which
// run_pre_init already constructed (running it again would wipe what nninitStartup has set up in it).
static bool entry_calls(uint64_t entry, uint64_t target) {
    if (!target) return false;
    for (int i = 0; i < 24; i++) {
        const uint64_t pc = entry + 4 * uint64_t(i);
        const uint32_t w = ld<uint32_t>(pc);
        if ((w & 0x7C000000) == 0x14000000) {  // b / bl
            const int64_t off = int64_t(int32_t(w << 6) >> 6) * 4;
            if (pc + uint64_t(off) == target) return true;
        }
        if (w == 0xD65F03C0) break;  // ret
    }
    return false;
}

static void run_init_array(CPU& c, const ModuleInfo& m) {
    if (!m.init_array) {
        if (m.init_count) log("init_module %s: init_array is null but has %zu entries: constructors skipped", m.name, size_t(m.init_count));
        return;
    }
    const uint64_t os_ctor = module_symbol(m, "_ZN2nn2os6detail17OsResourceManagerC2Ev");
    size_t ran = 0, left_out = 0;
    for (size_t k = 0; k < m.init_count; k++) {
        const uint64_t entry = m.base + m.init_array[k];
        if (entry_calls(entry, os_ctor)) {
            log("init_module %s: init_array[%zu] at %s builds the OS resource manager, already constructed: left out", m.name,
                k, describe(entry).c_str());
            left_out++;
            continue;
        }
        call_preserving(c, entry);
        ran++;
    }
    log("init_module %s: %zu constructors run, %zu left out", m.name, ran, left_out);
}

static void init_module(CPU& c, const ModuleInfo& m) {
    bool from_symbol = false;
    const uint64_t dt_init_at = dt_init_addr(m, &from_symbol);
    const uint32_t dt_init = ld<uint32_t>(dt_init_at);
    log("init_module %s: DT_INIT at %llx (%s) = %08x, init_array %p with %zu entries", m.name,
        (unsigned long long)dt_init_at, from_symbol ? "the _init symbol" : "default +0x30", dt_init,
        (const void*)m.init_array, size_t(m.init_count));
    const char* skip = std::getenv("NX_SKIP_DT_INIT");
    const bool skip_dt_init = skip && skip[0] == '1';
    if (skip_dt_init || !dt_init_usable(dt_init)) {
        // NX_SKIP_DT_INIT=1, or no code there: DT_INIT is not called, so the libc registration it does is not done
        // and no status can be read. The constructors are run directly only when asked (NX_FORCE_CTORS=1), as
        // nninitInitializeSdkModule may already have run them.
        log("init_module %s: DT_INIT skipped (%s)", m.name, skip_dt_init ? "NX_SKIP_DT_INIT=1" : "not code");
        const char* force = std::getenv("NX_FORCE_CTORS");
        if (skip_dt_init || (force && force[0] == '1')) run_init_array(c, m);
        return;
    }
    // DT_INIT keeps the result of its registration in a status word; read it from the code (no call needed). A
    // nonzero word means DT_INIT already ran (nninitInitializeSdkModule can do it): it is not called a second time.
    const uint64_t status_at = init_status_addr(m, dt_init_at, false);
    const uint32_t before = status_at ? ld<uint32_t>(status_at) : 0;
    log("init_module %s: init status word at %llx = %u before DT_INIT", m.name, (unsigned long long)status_at, before);
    if (before == 0) {
        log("init_module %s: calling DT_INIT at %s", m.name, describe(dt_init_at).c_str());
        call_preserving(c, dt_init_at);
        log("init_module %s: DT_INIT returned", m.name);
    } else {
        log("init_module %s: DT_INIT already ran: not called again", m.name);
    }
    const uint32_t status = ld<uint32_t>(init_status_addr(m, dt_init_at));
    log("init_module %s: init status %u", m.name, status);
    if (status == 1) run_init_array(c, m);
    else log("%s constructors: 0 (status %u)", m.name, status);
}

static void rtld_init(CPU&) {}
static void rtld_fini(CPU&) {}
static void rtld_mark(CPU&) {}

// The address of a symbol the module itself defines (not an import), or 0.
static uint64_t module_symbol(const ModuleInfo& m, const char* name) {
    for (size_t k = 0; k < m.symbol_count; k++)
        if (!m.symbols[k].imported && m.symbols[k].name && std::string(m.symbols[k].name) == name)
            return m.base + m.symbols[k].value;
    return 0;
}

// For a game whose nn::init::Start cannot be found: the first `limit` symbols the module defines, then every defined
// symbol whose name looks like a start function (Start, start, nnMain, nninit, _init), so the real name shows in nx.log.
static void log_symbols(const ModuleInfo& m, const char* label, size_t limit) {
    size_t defined = 0;
    for (size_t k = 0; k < m.symbol_count; k++)
        if (!m.symbols[k].imported) defined++;
    log("symbols of %s: %zu in total, %zu defined, base %llx", label, size_t(m.symbol_count), defined,
        (unsigned long long)m.base);
    size_t shown = 0;
    for (size_t k = 0; k < m.symbol_count && shown < limit; k++) {
        if (m.symbols[k].imported) continue;
        log("  %s[%zu] %.160s at %llx", label, shown, m.symbols[k].name ? m.symbols[k].name : "(no name)",
            (unsigned long long)(m.base + m.symbols[k].value));
        shown++;
    }
    size_t hits = 0;
    for (size_t k = 0; k < m.symbol_count && hits < 20; k++) {
        if (m.symbols[k].imported || !m.symbols[k].name) continue;
        const std::string n = m.symbols[k].name;
        if (n.find("Start") == std::string::npos && n.find("start") == std::string::npos &&
            n.find("nnMain") == std::string::npos && n.find("nninit") == std::string::npos &&
            n.find("_init") == std::string::npos)
            continue;
        log("  %s start-like: %.160s at %llx", label, n.c_str(), (unsigned long long)(m.base + m.symbols[k].value));
        hits++;
    }
}

// First defined symbol of the module whose name contains `needle` (the address, 0 if none; the name in *name).
static uint64_t symbol_containing(const ModuleInfo& m, const char* needle, std::string* name) {
    for (size_t k = 0; k < m.symbol_count; k++) {
        if (m.symbols[k].imported || !m.symbols[k].name) continue;
        if (std::string(m.symbols[k].name).find(needle) == std::string::npos) continue;
        if (name) *name = m.symbols[k].name;
        return m.base + m.symbols[k].value;
    }
    return 0;
}

// A symbol of the sdk module first, then of the main module.
static uint64_t symbol_any(const char* name) {
    if (uint64_t a = module_symbol(*g_sdk, name)) return a;
    return module_symbol(*g_main, name);
}

static size_t count_of(const std::string& s, const char* needle) {
    size_t n = 0, at = 0;
    while ((at = s.find(needle, at)) != std::string::npos) n++, at += std::strlen(needle);
    return n;
}

// Names worth reading when the start sequence of an older SDK is unknown: anything of nn::init, and the nn::os
// initializers and resource manager (what a missing nn::os::Initialize leaves null). Long C++ names are cut.
static void log_candidates(const ModuleInfo& m, const char* label) {
    size_t shown = 0;
    for (size_t k = 0; k < m.symbol_count && shown < 40; k++) {
        if (m.symbols[k].imported || !m.symbols[k].name) continue;
        const std::string n = m.symbols[k].name;
        const bool os_init = n.find("nn2os") != std::string::npos && n.find("Initialize") != std::string::npos;
        if (n.find("nn4init") == std::string::npos && !os_init && n.find("ResourceManager") == std::string::npos &&
            n.find("Startup") == std::string::npos && n.find("InitializeSdkModule") == std::string::npos)
            continue;
        log("  %s candidate: %.160s at %llx", label, n.c_str(), (unsigned long long)(m.base + m.symbols[k].value));
        shown++;
    }
    log("  %s: %zu candidates listed", label, shown);
}

// The global storage object of a C++ constructor symbol, found by name: _ZN2nn2os6detail17OsResourceManagerC2Ev ->
// the class OsResourceManager -> the symbol containing "g_OsResourceManagerStorage". 0 if the name is not a
// constructor or has no such storage.
static uint64_t ctor_storage(const std::string& ctor, std::string* storage_name) {
    if (ctor.compare(0, 3, "_ZN") != 0 || ctor.size() < 8) return 0;
    const std::string tail = ctor.substr(ctor.size() - 4);
    if (tail != "C2Ev" && tail != "C1Ev") return 0;
    std::string cls;  // the last <length><identifier> before the constructor
    size_t i = 3;
    while (i < ctor.size() - 4 && ctor[i] >= '0' && ctor[i] <= '9') {
        size_t len = 0;
        while (i < ctor.size() && ctor[i] >= '0' && ctor[i] <= '9') len = len * 10 + size_t(ctor[i++] - '0');
        if (len == 0 || i + len > ctor.size() - 4) break;
        cls = ctor.substr(i, len);
        i += len;
    }
    if (cls.empty()) return 0;
    const std::string needle = "g_" + cls + "Storage";
    uint64_t a = symbol_containing(*g_sdk, needle.c_str(), storage_name);
    if (!a) a = symbol_containing(*g_main, needle.c_str(), storage_name);
    return a;
}

// Guest functions that set up the SDK before nninitStartup, in this order. Default: the OS resource manager's
// constructor, run on its global storage (g_OsResourceManagerStorage, found by name: x0 = the storage address), then
// the main module's nninitInitializeSdkModule. NX_PRE_INIT (config.txt) replaces the list: comma separated symbol
// names of the sdk or main module ("name@storage" gives the storage symbol for x0; a constructor name alone finds it),
// or "none".
static void run_pre_init(CPU& c) {
    std::vector<std::string> names;
    const char* env = std::getenv("NX_PRE_INIT");
    if (env && *env) {
        std::string list = env, item;
        for (size_t i = 0; i <= list.size(); i++) {
            if (i == list.size() || list[i] == ',') {
                if (!item.empty() && item != "none") names.push_back(item);
                item.clear();
            } else if (list[i] != ' ') {
                item += list[i];
            }
        }
    } else {
        names = {"_ZN2nn2os6detail17OsResourceManagerC2Ev", "nninitInitializeSdkModule"};
    }
    for (const std::string& entry : names) {
        std::string name = entry, storage_name;
        uint64_t arg0 = 0;
        const size_t at = entry.find('@');
        if (at != std::string::npos) {
            name = entry.substr(0, at);
            storage_name = entry.substr(at + 1);
            arg0 = symbol_any(storage_name.c_str());
            if (!arg0) {
                log("pre-init: storage %.160s not found, %.160s skipped", storage_name.c_str(), name.c_str());
                continue;
            }
        } else {
            arg0 = ctor_storage(name, &storage_name);
        }
        const uint64_t addr = symbol_any(name.c_str());
        if (!addr) {
            log("pre-init: %.160s not found, skipped", name.c_str());
            continue;
        }
        const bool is_ctor = name.size() > 4 && (name.compare(name.size() - 4, 4, "C2Ev") == 0 ||
                                                 name.compare(name.size() - 4, 4, "C1Ev") == 0);
        if (is_ctor && !arg0) {
            // A constructor with x0 = 0 only faults: no storage to build the object in.
            log("pre-init: no storage found for the constructor %.160s, skipped", name.c_str());
            continue;
        }
        for (int r = 0; r < 8; r++) c.x[r] = 0;
        c.x[0] = arg0;
        c.x[30] = hle_address("__process_return");
        if (arg0)
            log("pre-init: calling %.160s at %s, x0 = %llx (%.160s)", name.c_str(), describe(addr).c_str(),
                (unsigned long long)arg0, storage_name.c_str());
        else
            log("pre-init: calling %.160s at %s", name.c_str(), describe(addr).c_str());
        call(c, addr);
        log("pre-init: %.160s returned (x0 = %llx)", name.c_str(), (unsigned long long)c.x[0]);
    }
}

// nnMain, as called by nn::init::Start: the constructors need the program's allocator, which nninitStartup has
// set up by now, so they run here, right before the real nnMain.
static void nn_main(CPU& c) {
    log("nn_main: initializing the sdk module");
    init_module(c, *g_sdk);
    log("nn_main: sdk module initialized, initializing the main module");
    init_module(c, *g_main);
    log("nn_main: main module initialized, game::after_init()");
    game::after_init();
    log("nn_main: after_init done");
    const uint64_t real = module_symbol(*g_main, "nnMain");
    if (!real) fatal("nnMain not found in the main module");
    log("entering nnMain at %s", describe(real).c_str());
    call(c, real);
}

// longjmp(buf, val): unwinds the host stack to the recompiled function that called setjmp on buf.
static void long_jmp(CPU& c) { throw LongJmp{c.x[0], uint32_t(c.x[1])}; }

static void process_return(CPU&) {
    log("nn::init::Start returned");
    plat::exit_process(0);
}

NX_HLE("nnMain", nn_main);
NX_HLE("longjmp", long_jmp);
NX_HLE("_longjmp", long_jmp);
NX_HLE("__rtld_init", rtld_init);
NX_HLE("__rtld_fini", rtld_fini);
NX_HLE("__rtld_mark", rtld_mark);
NX_HLE("__process_return", process_return);

// ---- boot -----------------------------------------------------------------------------------------------------

void boot() {
    if (const char* t = std::getenv("NX_TRACE")) g_trace = *t == '1';
    if (plat::setting("trace") == "1") g_trace = true;
    memory_init();
    kernel_init();
#if NX_NCE
    // The player's own ExeFS (main and sdk as they are on the cartridge or in the eShop download): from the game file
    // read in place (game_image.h), else from the data folder's exefs/.
    nce::init();
    static std::vector<std::unique_ptr<nce::NsoModule>> nso;
    nso.clear();
    const std::string exefs = plat::data_root() + "/exefs";
    const nxi::GameImage* image = game_image();

    auto find_mod_exefs = [](const std::string& name) -> std::string {
        std::error_code ec;
        fs::path root = fs::u8path(plat::data_root()) / "mods";
        if (fs::is_regular_file(root / "exefs" / fs::u8path(name), ec))
            return (root / "exefs" / fs::u8path(name)).string();
        if (fs::is_directory(root, ec)) {
            for (auto& e : fs::directory_iterator(root, ec)) {
                if (e.is_directory(ec)) {
                    fs::path candidate = e.path() / "exefs" / fs::u8path(name);
                    if (fs::is_regular_file(candidate, ec)) return candidate.string();
                }
            }
        }
        return {};
    };

    std::vector<std::string> mod_names = {"main", "sdk"};
    for (int k = 0; k <= 9; k++) mod_names.push_back("subsdk" + std::to_string(k));

    for (const auto& name : mod_names) {
        std::string mod_file = find_mod_exefs(name);
        std::string error;
        const uint64_t base = kModuleBase + uint64_t(nso.size()) * 0x10000000;
        if (!mod_file.empty()) {
            log("nce: loading mod exefs override: %s", mod_file.c_str());
            auto m = nce::load_nso(mod_file, name, base, error);
            if (m) nso.push_back(std::move(m));
            else log("nce: error loading mod %s: %s", mod_file.c_str(), error.c_str());
        } else if (name == "main" || name == "sdk") {
            // Only main and sdk come from base game. Base game subsdk0 is the Switch Tegra NVN driver and must not be loaded.
            if (image) {
                auto it = image->exefs().find(name);
                if (it != image->exefs().end()) {
                    auto m = nce::load_nso_bytes(it->second, std::string("ExeFS/") + name, name, base, error);
                    if (m) nso.push_back(std::move(m));
                } else {
                    fatal("the game file's ExeFS has no %s", name.c_str());
                }
            } else {
                std::error_code ec;
                if (fs::is_regular_file(fs::u8path(exefs) / fs::u8path(name), ec)) {
                    auto m = nce::load_nso(exefs + "/" + name, name, base, error);
                    if (m) nso.push_back(std::move(m));
                } else {
                    fatal("exefs/%s not found (copy the exefs folder of your own dump there)", name.c_str());
                }
            }
        }
    }
    if (nso.size() < 2) fatal("failed to load base modules (main and sdk required)");

    if (const char* want = game::kInfo.main_build_id) {
        std::string have;
        for (uint8_t b : nso[0]->build_id) {
            char hex[3];
            std::snprintf(hex, sizeof hex, "%02X", b);
            have += hex;
        }
        // Informational only: the Title ID of the game this fork is being adapted to. main.cpp never checked a
        // Title ID; the only gate here is the main NSO build ID below.
        constexpr const char* kFarmingSimTitleId = "0100D04004176000";  // Farming Simulator NSE, v1.5.0.1
        log("target title id %s, main build id %s", kFarmingSimTitleId, have.c_str());

        const char* known_builds[] = {
            "708BCACA873EA32B", "B78A03E489E5BD92", "6E1F4CAE92A28C91", "2F2253ACD6BE5D87"
        };
        bool known = false;
        for (const char* k : known_builds) {
            if (have.compare(0, std::strlen(k), k) == 0) { known = true; break; }
        }
        // Bypass enabled by default. NX_ALLOW_ANY_BUILD=0 re-enables the strict check.
        const char* strict = std::getenv("NX_ALLOW_ANY_BUILD");
        bool allow_any = !(strict && *strict == '0');
        if (known) {
            log("main build %s recognized (Persona 5 Royal)", have.c_str());
        } else if (allow_any) {
            log("main build %s accepted: build check bypassed", have.c_str());
        } else {
            fatal("exefs/main is build %s, not %s. Remove NX_ALLOW_ANY_BUILD=0 to bypass.",
                  have.c_str(), want);
        }
    }

    g_main = &nso[0]->info, g_sdk = &nso[1]->info;
    std::vector<const ModuleInfo*> mods;
    std::vector<const uint8_t*> img_ptrs;
    for (const auto& m : nso) {
        mods.push_back(&m->info);
        img_ptrs.push_back(m->image.data());
    }
    load_modules(mods.data(), mods.size(), "", img_ptrs.data());
#else
    g_main = &nx_module_main, g_sdk = &nx_module_sdk;
    const ModuleInfo* mods[] = {g_main, g_sdk};
    load_modules(mods, 2, plat::data_root() + "/analysis");
#endif

    uint64_t tdata = 0, tdata_end = 0;
    if (tdata_range(*g_main, tdata, tdata_end) && tdata_end > tdata) {
        set_tls_template(tdata, tdata_end);
        log("main .tdata: %llu bytes", (unsigned long long)(tdata_end - tdata));
    }
    uint32_t main_thread = create_main_thread(16ull << 20);
    CPU& c = current_cpu();
    c.x[0] = main_thread;
    c.x[1] = g_main->base;
    // SDK 9: Start(thread, main base, init, fini). SDK 16 adds a first callback: rtld's sets a word to 1 (rtld+0x1330
    // in Wonder), before its init and fini that walk the module list.
    uint64_t start = find_symbol("_ZN2nn4init5StartEmmPFvvES2_");
    if (start) {
        c.x[2] = hle_address("__rtld_init");
        c.x[3] = hle_address("__rtld_fini");
    } else if ((start = find_symbol("_ZN2nn4init5StartEmmPFvvES2_S2_"))) {
        c.x[2] = hle_address("__rtld_mark");
        c.x[3] = hle_address("__rtld_init");
        c.x[4] = hle_address("__rtld_fini");
    } else {
        log("nn::init::Start not found (neither the SDK 9 nor the SDK 16 signature): looking for another way in");
        log_symbols(*g_sdk, "sdk", 20);
        log_symbols(*g_main, "main", 20);
        log_candidates(*g_sdk, "sdk");
        log_candidates(*g_main, "main");
        c.x[30] = hle_address("__process_return");
        std::string alt_name;
        uint64_t alt = symbol_containing(*g_sdk, "nn4init5Start", &alt_name);
        if (!alt) alt = symbol_containing(*g_main, "nn4init5Start", &alt_name);
        const uint64_t real_main = module_symbol(*g_main, "nnMain");
        if (alt) {
            // Fallback 0: nn::init::Start with another signature. The number of callbacks comes from the mangled
            // name: one function pointer type, then a substitution (S2_) for each further one.
            const size_t callbacks = count_of(alt_name, "PFvvE") + count_of(alt_name, "S2_");
            log("fallback 0: %.160s at %s, %zu callbacks", alt_name.c_str(), describe(alt).c_str(), callbacks);
            start = alt;
            if (callbacks >= 3) {
                c.x[2] = hle_address("__rtld_mark");
                c.x[3] = hle_address("__rtld_init");
                c.x[4] = hle_address("__rtld_fini");
            } else {
                c.x[2] = hle_address("__rtld_init");
                c.x[3] = hle_address("__rtld_fini");
            }
        } else if (real_main) {
            // Fallback 1: the game's own nnMain, after the SDK setup the missing Start would have done (see
            // run_pre_init: the OS resource manager constructor, the main module's nninitInitializeSdkModule), then nninitStartup
            // (the program's allocator and TLS), then nn_main (module constructors and nnMain).
            // Whatever else Start does (nn::diag, the main thread handle) is skipped: see the symbols above.
            const uint64_t startup = module_symbol(*g_main, "nninitStartup");
            log("fallback 1: nnMain at %s, nninitStartup %s", describe(real_main).c_str(),
                startup ? describe(startup).c_str() : "not found (skipped)");
            run_pre_init(c);
            if (startup) {
                for (int r = 0; r < 8; r++) c.x[r] = 0;
                c.x[30] = hle_address("__process_return");
                call(c, startup);
                log("nninitStartup returned");
            }
            c.x[0] = main_thread;
            c.x[30] = hle_address("__process_return");
            log("calling nn_main (main thread handle %u)", main_thread);
            nn_main(c);
            log("returned from nnMain");
            plat::exit_process(0);
            return;
        } else {
            // Fallback 2 (last resort, experimental): no nnMain symbol (stripped). Jump to the start of a module
            // image: the main module's by default, the sdk's with NX_ENTRY=sdk. The first instruction of a module's
            // text is the branch to its entry. The symbol lists above are what to read to find the real start.
            const char* pick = std::getenv("NX_ENTRY");
            const bool use_sdk = pick && std::string(pick) == "sdk";
            const ModuleInfo& entry_mod = use_sdk ? *g_sdk : *g_main;
            start = entry_mod.base;
            c.x[2] = hle_address("__rtld_init");
            c.x[3] = hle_address("__rtld_fini");
            log("fallback 2: no nnMain either; entering the %s module image at %s (set NX_ENTRY=%s to try the other)",
                use_sdk ? "sdk" : "main", describe(start).c_str(), use_sdk ? "main" : "sdk");
        }
    }
    c.x[30] = hle_address("__process_return");
    log("entering %s at %s", start == g_main->base || start == g_sdk->base ? "the module entry" : "nn::init::Start",
        describe(start).c_str());
    call(c, start);
    log("returned from nn::init::Start");
}

}  // namespace nx
