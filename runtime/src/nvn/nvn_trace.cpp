// NVN entry points (docs/NVN.md, phases N2-N3).
//
// The game links one NVN symbol, nvnBootstrapLoader, and asks it for nvnDeviceGetProcAddress; every other NVN
// function is then fetched by name, the same shape as vkGetInstanceProcAddr. Both are HLE here. For each name:
//   - one we implement (an NX_HLE with the NVN name): our function;
//   - a builder or state object proc (pure struct setters/getters): the SDK driver's own, recompiled (see nvn.h);
//   - anything else: a tracing stub that records the call and returns a neutral value, so the trace goes as far
//     as it can.
//   - a builder proc the SDK driver does not give (an SDK 2017 game: nvn::real_proc returns 0): the host's own
//     version below (MemoryPoolBuilder: kept in the builder's 64 bytes), instead of a stub that stores nothing.
// Every name the game asks for is logged to nx.log (once per name, with where it was resolved), and a null answer
// always is.
// The trace is <data root>/nvn_trace.txt: one line per call (the first kFullCalls of each proc), each proc's
// driver address ("# real"), and a count per proc at exit.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>

#include "nvn.h"
#include "platform.h"

namespace nx {

namespace {

constexpr uint32_t kFullCalls = 32;  // calls of one proc traced with their arguments

std::mutex g_lock;
FILE* g_trace_file = nullptr;
std::map<std::string, uint64_t> g_calls;  // proc -> calls
std::atomic<uint64_t> g_seq{0};

std::string guest_str(uint64_t addr) {
    if (!addr) return "(null)";
    return std::string(reinterpret_cast<const char*>(addr), strnlen(reinterpret_cast<const char*>(addr), 256));
}

void write_line(const char* line) {  // g_lock held
    if (!g_trace_file) {
        g_trace_file = plat::fopen_utf8(plat::data_root() + "/nvn_trace.txt", "w");
        std::atexit([] {
            std::lock_guard lk(g_lock);
            if (!g_trace_file) return;
            std::fprintf(g_trace_file, "\n# calls per proc\n");
            for (auto& [name, n] : g_calls) std::fprintf(g_trace_file, "%10llu %s\n", (unsigned long long)n, name.c_str());
            std::fclose(g_trace_file);
            g_trace_file = nullptr;
        });
    }
    if (g_trace_file) {
        std::fputs(line, g_trace_file);
        std::fputc('\n', g_trace_file);
        std::fflush(g_trace_file);
    }
}

// A proc we do not implement yet: record it, return 0 except for the ones whose 0 means "failed" to the game.
void trace_stub(CPU& c) {
    const char* name = hle_current_name();
    uint64_t n, seq = g_seq++;
    {
        std::lock_guard lk(g_lock);
        n = ++g_calls[name];
        if (n <= kFullCalls) {
            char line[512];
            std::snprintf(line, sizeof line, "%8llu %s(%llx, %llx, %llx, %llx, %llx, %llx) lr %s", (unsigned long long)seq,
                          name, (unsigned long long)c.x[0], (unsigned long long)c.x[1], (unsigned long long)c.x[2],
                          (unsigned long long)c.x[3], (unsigned long long)c.x[4], (unsigned long long)c.x[5],
                          describe(c.x[30]).c_str());
            write_line(line);
        }
    }
    if (n == 1) log("nvn: first call of %s (lr %s)", name, describe(c.x[30]).c_str());
    // Initialize functions return NVNboolean: false would make the engine give up at the first object.
    c.x[0] = std::strstr(name, "Initialize") ? 1 : 0;
}

// Builders, state objects and texture views only store and return fields: the driver's own code serves them.
bool driver_serves(const std::string& name) {
    static const std::regex pure(
        "nvn(\\w+Builder\\w+|(Blend|Color|DepthStencil|Multisample|Polygon|VertexAttrib|VertexStream|ChannelMask)"
        "State\\w+|TextureView\\w+)");
    return std::regex_match(name, pure) && !std::getenv("NX_NVN_NO_DRIVER");
}

// ---- host MemoryPoolBuilder ------------------------------------------------------------------------------------
// NVNmemoryPoolBuilder is 64 opaque bytes the game only touches through these procs, so the host can keep its own
// layout in them. Used when the SDK's driver has no proc for the name (nvn::real_proc returned 0).
struct HostPoolBuilder {
    uint64_t device;      // +0x00 NVNdevice*
    uint64_t memory;      // +0x08 void* storage
    uint64_t size;        // +0x10 storage size
    uint32_t flags;       // +0x18 NVNmemoryPoolFlags
    uint32_t reserved0;   // +0x1c
    uint64_t reserved1[3];  // +0x20
    uint64_t magic;       // +0x38 kPoolBuilderMagic: built by the host procs
};
static_assert(sizeof(HostPoolBuilder) == 64, "NVNmemoryPoolBuilder is 64 bytes");
constexpr uint64_t kPoolBuilderMagic = 0x42504d48;  // "HMPB"

HostPoolBuilder* pool_builder(CPU& c) {
    auto* b = reinterpret_cast<HostPoolBuilder*>(c.x[0]);
    if (b) b->magic = kPoolBuilderMagic;
    return b;
}

// void nvnMemoryPoolBuilderSetDefaults(NVNmemoryPoolBuilder*): no device, no storage, no flags (the game sets them).
void host_pool_builder_set_defaults(CPU& c) {
    if (auto* b = reinterpret_cast<HostPoolBuilder*>(c.x[0])) {
        std::memset(b, 0, sizeof *b);
        b->magic = kPoolBuilderMagic;
    }
}
// void nvnMemoryPoolBuilderSetDevice(NVNmemoryPoolBuilder*, NVNdevice*)
void host_pool_builder_set_device(CPU& c) {
    if (auto* b = pool_builder(c)) b->device = c.x[1];
}
// void nvnMemoryPoolBuilderSetFlags(NVNmemoryPoolBuilder*, int flags)
void host_pool_builder_set_flags(CPU& c) {
    if (auto* b = pool_builder(c)) b->flags = uint32_t(c.x[1]);
}
// void nvnMemoryPoolBuilderSetStorage(NVNmemoryPoolBuilder*, void* memory, size_t size)
void host_pool_builder_set_storage(CPU& c) {
    if (auto* b = pool_builder(c)) b->memory = c.x[1], b->size = c.x[2];
}
// const NVNdevice* nvnMemoryPoolBuilderGetDevice(const NVNmemoryPoolBuilder*)
void host_pool_builder_get_device(CPU& c) {
    auto* b = reinterpret_cast<HostPoolBuilder*>(c.x[0]);
    c.x[0] = b ? b->device : 0;
}
// void* nvnMemoryPoolBuilderGetMemory(const NVNmemoryPoolBuilder*)
void host_pool_builder_get_memory(CPU& c) {
    auto* b = reinterpret_cast<HostPoolBuilder*>(c.x[0]);
    c.x[0] = b ? b->memory : 0;
}
// size_t nvnMemoryPoolBuilderGetSize(const NVNmemoryPoolBuilder*)
void host_pool_builder_get_size(CPU& c) {
    auto* b = reinterpret_cast<HostPoolBuilder*>(c.x[0]);
    c.x[0] = b ? b->size : 0;
}
// NVNmemoryPoolFlags nvnMemoryPoolBuilderGetFlags(const NVNmemoryPoolBuilder*)
void host_pool_builder_get_flags(CPU& c) {
    auto* b = reinterpret_cast<HostPoolBuilder*>(c.x[0]);
    c.x[0] = b ? b->flags : 0;
}

// The host's version of a builder proc, registered below as "host_<nvn name>"; 0 when there is none.
uint64_t host_builder_proc(const std::string& name) {
    const std::string host = "host_" + name;
    return hle_lookup(host.c_str()) ? hle_address(host.c_str()) : 0;
}

// One line in nx.log for each name the game asks for (the first time), and for every null answer.
void note_proc(const char* who, uint64_t device, const std::string& name, const char* how, uint64_t addr) {
    static std::mutex m;
    static std::set<std::string> seen;
    bool first;
    {
        std::lock_guard lk(m);
        first = seen.insert(name).second;
    }
    if (!addr)
        log("nvn: %s(%llx, %s) -> 0 (%s): the game gets a null function pointer", who, (unsigned long long)device,
            name.c_str(), how);
    else if (first)
        log("nvn: %s(%llx, %s) -> %s at %s", who, (unsigned long long)device, name.c_str(), how, describe(addr).c_str());
}

uint64_t proc_address(CPU& c, const std::string& name, const char* who, uint64_t device) {
    static const bool loader_noted = [&] {
        const uint64_t loader = find_symbol("nvnBootstrapLoader");
        log("nvn: the sdk's own nvnBootstrapLoader (the driver): %s", loader ? describe(loader).c_str() : "not found");
        return true;
    }();
    (void)loader_noted;
    if (name.empty()) {
        note_proc(who, device, name, "empty name", 0);
        return 0;
    }
    static const bool note_all = std::getenv("NX_NVN_REAL") != nullptr;  // record every proc's driver address
    if (note_all) nvn::real_proc(c, name.c_str());
    uint64_t addr = 0;
    const char* how = "";
    if (hle_lookup(name.c_str())) {
        addr = hle_address(name.c_str());
        how = "host implementation";
    } else {
        if (driver_serves(name)) {
            // Logged BEFORE the call: the loader is guest code, and a crash inside it would otherwise leave no trace
            // of which name it was asked for.
            static std::mutex asked_lock;
            static std::set<std::string> asked;
            bool first;
            {
                std::lock_guard lk(asked_lock);
                first = asked.insert(name).second;
            }
            if (first) log("nvn: asking the driver's loader for %s", name.c_str());
            if (uint64_t real = nvn::real_proc(c, name.c_str())) {
                addr = real;
                how = "the driver's own proc";
            }
        }
        // No driver proc (an SDK 2017 game, or NX_NVN_NO_DRIVER=1): the host's own version, if there is one.
        if (!addr) {
            if (uint64_t host = host_builder_proc(name)) {
                addr = host;
                how = "host builder (no driver proc)";
            }
        }
    }
    if (!addr) {
        addr = hle_address_or(name.c_str(), trace_stub);
        how = driver_serves(name) ? "trace stub (the driver has no such proc, no host version)"
                                  : "trace stub (not implemented)";
    }
    note_proc(who, device, name, how, addr);
    return addr;
}

// PFNNVNGENERICFUNCPROC nvnBootstrapLoader(const char* name)
void nvn_bootstrap_loader(CPU& c) {
    std::string name = guest_str(c.x[0]);
    log("nvnBootstrapLoader(%s)", name.c_str());
    {
        std::lock_guard lk(g_lock);
        write_line(("# nvnBootstrapLoader(" + name + ")").c_str());
    }
    c.x[0] = proc_address(c, name, "nvnBootstrapLoader", 0);
}

// PFNNVNGENERICFUNCPROC nvnDeviceGetProcAddress(const NVNdevice* device, const char* name)
void nvn_device_get_proc_address(CPU& c) {
    std::string name = guest_str(c.x[1]);
    if (g_trace) log("nvnDeviceGetProcAddress(%llx, %s)", (unsigned long long)c.x[0], name.c_str());
    c.x[0] = proc_address(c, name, "nvnDeviceGetProcAddress", c.x[0]);
}

}  // namespace

NX_HLE("nvnBootstrapLoader", nvn_bootstrap_loader);
NX_HLE("nvnDeviceGetProcAddress", nvn_device_get_proc_address);

// Host versions of builder procs, used when the SDK's driver has none (see proc_address).
NX_HLE("host_nvnMemoryPoolBuilderSetDefaults", host_pool_builder_set_defaults);
NX_HLE("host_nvnMemoryPoolBuilderSetDevice", host_pool_builder_set_device);
NX_HLE("host_nvnMemoryPoolBuilderSetFlags", host_pool_builder_set_flags);
NX_HLE("host_nvnMemoryPoolBuilderSetStorage", host_pool_builder_set_storage);
NX_HLE("host_nvnMemoryPoolBuilderGetDevice", host_pool_builder_get_device);
NX_HLE("host_nvnMemoryPoolBuilderGetMemory", host_pool_builder_get_memory);
NX_HLE("host_nvnMemoryPoolBuilderGetSize", host_pool_builder_get_size);
NX_HLE("host_nvnMemoryPoolBuilderGetFlags", host_pool_builder_get_flags);

namespace nvn {

// What the game stored in a memory pool builder through the host procs above (nvnMemoryPoolInitialize reads the
// builder with this: declare it in nvn.h). False when the builder was not built by the host procs (the driver's own
// layout, which its getters read).
bool host_pool_builder(uint64_t builder, uint64_t& device, uint64_t& memory, uint64_t& size, uint32_t& flags) {
    const auto* b = reinterpret_cast<const HostPoolBuilder*>(builder);
    if (!b || b->magic != kPoolBuilderMagic) return false;
    device = b->device, memory = b->memory, size = b->size, flags = b->flags;
    return true;
}

uint64_t guest_call(CPU& c, uint64_t fn, std::initializer_list<uint64_t> args) {
    uint64_t saved[31];
    std::memcpy(saved, c.x, sizeof saved);
    size_t k = 0;
    for (uint64_t a : args) c.x[k++] = a;
    call(c, fn);
    uint64_t r = c.x[0];
    std::memcpy(c.x, saved, sizeof saved);
    return r;
}

float guest_call_f(CPU& c, uint64_t fn, std::initializer_list<uint64_t> args) {
    uint64_t saved[31];
    std::memcpy(saved, c.x, sizeof saved);
    V128 v0 = c.v[0];
    size_t k = 0;
    for (uint64_t a : args) c.x[k++] = a;
    call(c, fn);
    float r = c.v[0].f32[0];
    std::memcpy(c.x, saved, sizeof saved);
    c.v[0] = v0;
    return r;
}

uint64_t scratch(CPU&, size_t size) {
    thread_local uint64_t page = 0;
    if (!page) page = kernel_alloc(0x1000);
    if (size > 0x1000) fatal("nvn scratch: %zu bytes asked", size);
    return page;
}

uint64_t real_proc_slow(CPU& c, const char* name);

uint64_t real_proc(CPU& c, const char* name) {
    // The hot callers (state-object getters, one per bound state) pass string literals: a per-thread cache by the
    // literal's address, verified by content, instead of a global mutex and a string-keyed tree at every call (the
    // game's recording threads contended on that lock).
    struct Hit {
        std::string name;
        uint64_t proc = 0;
    };
    thread_local std::unordered_map<const char*, Hit> fast;
    if (auto it = fast.find(name); it != fast.end() && it->second.name == name) return it->second.proc;
    uint64_t found = real_proc_slow(c, name);
    if (found) fast[name] = {name, found};
    return found;
}

uint64_t real_proc_slow(CPU& c, const char* name) {
    static std::mutex lock;
    static std::map<std::string, uint64_t> cache;
    {
        std::lock_guard lk(lock);
        auto it = cache.find(name);
        if (it != cache.end()) return it->second;
    }
    static uint64_t loader = find_symbol("nvnBootstrapLoader");  // the sdk's export (imports are HLE)
    static uint64_t scratch = kernel_alloc(0x1000);
    uint64_t real = 0;
    size_t len = std::strlen(name);
    if (loader && len < 0x1000) {
        std::lock_guard lk(lock);  // one scratch buffer
        std::memcpy(reinterpret_cast<void*>(scratch), name, len + 1);
        real = guest_call(c, loader, {scratch});
        cache[name] = real;
    }
    {
        std::lock_guard lk(g_lock);
        write_line(("# real " + std::string(name) + " = " + describe(real)).c_str());
    }
    return real;
}

ObjectsLock& objects_lock() {
    static ObjectsLock m;
    return m;
}

std::vector<std::pair<uint64_t, std::unique_ptr<Object>>>& deferred_objects() {  // (epoch when retired, object)
    static std::vector<std::pair<uint64_t, std::unique_ptr<Object>>> v;
    return v;
}

void defer_object(std::unique_ptr<Object> o) { deferred_objects().push_back({current_epoch(), std::move(o)}); }

void drain_objects() {
    // Only what no replay can name any more: retired at or before the last replay's mark (nvn_incarnation.cpp). A
    // later submit's lists were recorded after that, so they cannot mean those objects.
    std::vector<std::unique_ptr<Object>> dead;
    {
        std::lock_guard lk(objects_lock());
        uint64_t upto = replay_floor();
        auto& all = deferred_objects();
        size_t keep = 0;
        for (auto& [epoch, o] : all) {
            if (epoch <= upto)
                dead.push_back(std::move(o));
            else
                all[keep++] = {epoch, std::move(o)};
        }
        all.resize(keep);
        purge_incarnations(upto);  // the records point at objects freed here
    }
}  // destructors run here, outside the lock (a Texture's takes the graveyard lock)

std::unordered_map<uint64_t, std::unique_ptr<Object>>& objects() {
    static std::unordered_map<uint64_t, std::unique_ptr<Object>> m;
    return m;
}

}  // namespace nvn

}  // namespace nx
