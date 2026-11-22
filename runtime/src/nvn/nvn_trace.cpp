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
#include <unordered_map>

#include "nvn.h"
#include "platform.h"

namespace nx {

namespace nvn {
// The SDK driver's own proc for a name (its loader, recompiled in sdk): 0 when the sdk has no driver.
uint64_t driver_proc(CPU& c, const char* name);
}  // namespace nvn

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

// ---- host builders: every other builder, kept in a side table keyed by the builder's address ---------------------
// The builders are opaque structs only these procs touch, so the host keeps its own copy of the fields; a side table
// (not the builder's bytes) because the real sizes of the structs differ by builder and version. SetDefaults clears an
// entry. The values marked "assumed" are NVN defaults this code has not checked against a driver.
struct BuilderData {
    uint64_t f[64];
};
std::mutex g_bd_lock;
std::unordered_map<uint64_t, BuilderData> g_bd;

BuilderData* bd(uint64_t builder) { return builder ? &g_bd[builder] : nullptr; }  // g_bd_lock held

uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float float_of(uint64_t u) {
    uint32_t v = uint32_t(u);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}
void put32(uint64_t addr, uint32_t v) {
    if (addr) std::memcpy(reinterpret_cast<void*>(addr), &v, 4);
}
void put8(uint64_t addr, uint8_t v) {
    if (addr) std::memcpy(reinterpret_cast<void*>(addr), &v, 1);
}
void put64(uint64_t addr, uint64_t v) {
    if (addr) std::memcpy(reinterpret_cast<void*>(addr), &v, 8);
}

// Queue builder defaults (assumed): command, compute and control memory, and the flush threshold.
constexpr uint64_t kQueueCommandMemory = 0x40000;
constexpr uint64_t kQueueComputeMemory = 0x40000;
constexpr uint64_t kQueueControlMemory = 0x1000;
constexpr uint64_t kQueueFlushThreshold = 0x2000;

void hb_DeviceBuilderSetDefaults(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    std::memset(d, 0, sizeof *d);
}

void hb_DeviceBuilderSetFlags(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[1] = uint32_t(c.x[1]);
}

void hb_QueueBuilderSetDefaults(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    std::memset(d, 0, sizeof *d);
    d->f[2] = kQueueCommandMemory;
    d->f[3] = kQueueComputeMemory;
    d->f[4] = kQueueControlMemory;
    d->f[5] = kQueueFlushThreshold;
}

void hb_QueueBuilderSetDevice(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[0] = c.x[1];
}

void hb_QueueBuilderSetFlags(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[1] = uint32_t(c.x[1]);
}

void hb_QueueBuilderSetCommandMemorySize(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[2] = c.x[1];
}

void hb_QueueBuilderSetComputeMemorySize(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = c.x[1];
}

void hb_QueueBuilderSetControlMemorySize(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[4] = c.x[1];
}

void hb_QueueBuilderSetCommandFlushThreshold(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[5] = c.x[1];
}

void hb_QueueBuilderSetQueueMemory(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[6] = c.x[1], d->f[7] = c.x[2];
}

void hb_QueueBuilderGetQueueMemorySize(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    const uint64_t sum = d ? d->f[2] + d->f[3] + d->f[4] : 0;
    c.x[0] = (sum + 0xfff) & ~uint64_t(0xfff);
}

void hb_WindowBuilderSetDefaults(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    std::memset(d, 0, sizeof *d);
    d->f[4] = 1;  // present interval
}

void hb_WindowBuilderSetDevice(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[0] = c.x[1];
}

void hb_WindowBuilderSetNativeWindow(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[1] = c.x[1];
}

void hb_WindowBuilderSetTextures(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = uint32_t(c.x[1]), d->f[2] = c.x[2];
}

void hb_WindowBuilderSetPresentInterval(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[4] = uint32_t(c.x[1]);
}

void hb_WindowBuilderGetNativeWindow(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1] : 0;
}

void hb_WindowBuilderGetPresentInterval(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[4] : 0;
}

void hb_BufferBuilderSetDefaults(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    std::memset(d, 0, sizeof *d);
}

void hb_BufferBuilderSetDevice(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[0] = c.x[1];
}

void hb_BufferBuilderSetStorage(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[1] = c.x[1], d->f[2] = c.x[2], d->f[3] = c.x[3];
}

void hb_BufferBuilderGetMemoryPool(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1] : 0;
}

void hb_BufferBuilderGetMemoryOffset(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2] : 0;
}

void hb_BufferBuilderGetSize(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[3] : 0;
}

void hb_SamplerBuilderSetDefaults(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    std::memset(d, 0, sizeof *d);
    d->f[7] = bits_of(1000.0f);  // lod clamp max
    d->f[11] = bits_of(1.0f);    // max anisotropy
}

void hb_SamplerBuilderSetDevice(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[0] = c.x[1];
}

void hb_SamplerBuilderSetMinMagFilter(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[1] = uint32_t(c.x[1]), d->f[2] = uint32_t(c.x[2]);
}

void hb_SamplerBuilderSetWrapMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = uint32_t(c.x[1]), d->f[4] = uint32_t(c.x[2]), d->f[5] = uint32_t(c.x[3]);
}

void hb_SamplerBuilderSetLodClamp(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[6] = bits_of(c.v[0].f32[0]), d->f[7] = bits_of(c.v[1].f32[0]);
}

void hb_SamplerBuilderSetLodBias(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[8] = bits_of(c.v[0].f32[0]);
}

void hb_SamplerBuilderSetCompare(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[9] = uint32_t(c.x[1]), d->f[10] = uint32_t(c.x[2]);
}

void hb_SamplerBuilderSetBorderColor(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !c.x[1]) return;
    std::memcpy(&d->f[13], reinterpret_cast<const void*>(c.x[1]), 16);  // the 4 components, raw
}

void hb_SamplerBuilderGetBorderColor(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !c.x[1]) return;
    std::memcpy(reinterpret_cast<void*>(c.x[1]), &d->f[13], 16);
}

void hb_SamplerBuilderSetBorderColori(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !c.x[1]) return;
    std::memcpy(&d->f[13], reinterpret_cast<const void*>(c.x[1]), 16);  // the 4 components, raw
}

void hb_SamplerBuilderGetBorderColori(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !c.x[1]) return;
    std::memcpy(reinterpret_cast<void*>(c.x[1]), &d->f[13], 16);
}

void hb_SamplerBuilderSetBorderColorui(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !c.x[1]) return;
    std::memcpy(&d->f[13], reinterpret_cast<const void*>(c.x[1]), 16);  // the 4 components, raw
}

void hb_SamplerBuilderGetBorderColorui(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !c.x[1]) return;
    std::memcpy(reinterpret_cast<void*>(c.x[1]), &d->f[13], 16);
}

void hb_SamplerBuilderSetMaxAnisotropy(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[11] = bits_of(c.v[0].f32[0]);
}

void hb_SamplerBuilderSetReductionFilter(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[12] = uint32_t(c.x[1]);
}

void hb_SamplerBuilderGetMinMagFilter(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) put32(c.x[1], uint32_t(d->f[1])), put32(c.x[2], uint32_t(d->f[2]));
}

void hb_SamplerBuilderGetWrapMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) put32(c.x[1], uint32_t(d->f[3])), put32(c.x[2], uint32_t(d->f[4])), put32(c.x[3], uint32_t(d->f[5]));
}

void hb_SamplerBuilderGetLodClamp(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) put32(c.x[1], uint32_t(d->f[6])), put32(c.x[2], uint32_t(d->f[7]));
}

void hb_SamplerBuilderGetLodBias(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.v[0].f32[0] = d ? float_of(d->f[8]) : 0.0f;
}

void hb_SamplerBuilderGetCompare(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) put32(c.x[1], uint32_t(d->f[9])), put32(c.x[2], uint32_t(d->f[10]));
}

void hb_SamplerBuilderGetMaxAnisotropy(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.v[0].f32[0] = d ? float_of(d->f[11]) : 1.0f;
}

void hb_SamplerBuilderGetReductionFilter(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[12] : 0;
}

void hb_TextureBuilderSetDefaults(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    std::memset(d, 0, sizeof *d);
    d->f[4] = 1;  // height
    d->f[5] = 1;  // depth
    d->f[6] = 1;  // levels
}

void hb_TextureBuilderSetDevice(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[0] = c.x[1];
}

void hb_TextureBuilderSetFlags(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[1] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetTarget(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[2] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetWidth(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetHeight(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[4] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetDepth(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[5] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetSize1D(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = uint32_t(c.x[1]), d->f[4] = 1, d->f[5] = 1;
}

void hb_TextureBuilderSetSize2D(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = uint32_t(c.x[1]), d->f[4] = uint32_t(c.x[2]), d->f[5] = 1;
}

void hb_TextureBuilderSetSize3D(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[3] = uint32_t(c.x[1]), d->f[4] = uint32_t(c.x[2]), d->f[5] = uint32_t(c.x[3]);
}

void hb_TextureBuilderSetLevels(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[6] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetFormat(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[7] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetSamples(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[8] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetSwizzle(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0]))
        d->f[9] = (c.x[1] & 0xff) | (c.x[2] & 0xff) << 8 | (c.x[3] & 0xff) << 16 | (c.x[4] & 0xff) << 24;
}

void hb_TextureBuilderSetDepthStencilMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[10] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetStorage(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[13] = c.x[1], d->f[14] = c.x[2];
}

void hb_TextureBuilderSetStride(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[11] = c.x[1];
}

void hb_TextureBuilderSetGLTextureName(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[12] = uint32_t(c.x[1]);
}

void hb_TextureBuilderSetPackagedTextureData(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[15] = c.x[1];
}

void hb_TextureBuilderSetPackagedTextureLayout(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) d->f[16] = c.x[1];  // the pointer the game gave (its struct is opaque here)
}

void hb_TextureBuilderGetFlags(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1] : 0;
}

void hb_TextureBuilderGetTarget(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2] : 0;
}

void hb_TextureBuilderGetWidth(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[3] : 0;
}

void hb_TextureBuilderGetHeight(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[4] : 0;
}

void hb_TextureBuilderGetDepth(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[5] : 0;
}

void hb_TextureBuilderGetLevels(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[6] : 0;
}

void hb_TextureBuilderGetFormat(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[7] : 0;
}

void hb_TextureBuilderGetSamples(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[8] : 0;
}

void hb_TextureBuilderGetDepthStencilMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[10] : 0;
}

void hb_TextureBuilderGetStride(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[11] : 0;
}

void hb_TextureBuilderGetGLTextureName(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[12] : 0;
}

void hb_TextureBuilderGetMemoryPool(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[13] : 0;
}

void hb_TextureBuilderGetMemoryOffset(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[14] : 0;
}

void hb_TextureBuilderGetPackagedTextureData(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[15] : 0;
}

void hb_TextureBuilderGetSwizzle(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    put32(c.x[1], uint32_t(d->f[9] & 0xff));
    put32(c.x[2], uint32_t(d->f[9] >> 8 & 0xff));
    put32(c.x[3], uint32_t(d->f[9] >> 16 & 0xff));
    put32(c.x[4], uint32_t(d->f[9] >> 24 & 0xff));
}

// ---- host state objects and texture views: the same side table. A Get returns what a Set stored (a state's fields are
// the arguments of its setters; a keyed field (colour target index, face) is stored per key). SetDefaults clears all,
// so a field the game never sets reads as 0: the numeric value of an NVN enum's default is not known here (assumed).
// The setters/getters follow the NVN API as the author remembers it: see the notes in the reply.
void hd_BlendState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_BlendStateBlendTarget(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = uint32_t(c.x[1]);
}

void hg_BlendStateBlendTarget(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[0 + 0] : 0;
}

void hs_BlendStateBlendFunc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[1] = uint32_t(c.x[1]);
    d->f[2] = uint32_t(c.x[2]);
    d->f[3] = uint32_t(c.x[3]);
    d->f[4] = uint32_t(c.x[4]);
}

void hg_BlendStateBlendFunc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    put32(c.x[1], uint32_t(d->f[1 + 0]));
    put32(c.x[2], uint32_t(d->f[1 + 1]));
    put32(c.x[3], uint32_t(d->f[1 + 2]));
    put32(c.x[4], uint32_t(d->f[1 + 3]));
}

void hs_BlendStateBlendEquation(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[5] = uint32_t(c.x[1]);
    d->f[6] = uint32_t(c.x[2]);
}

void hg_BlendStateBlendEquation(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    put32(c.x[1], uint32_t(d->f[5 + 0]));
    put32(c.x[2], uint32_t(d->f[5 + 1]));
}

void hs_BlendStateAdvancedMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[7] = uint32_t(c.x[1]);
}

void hg_BlendStateAdvancedMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[7 + 0] : 0;
}

void hs_BlendStateAdvancedOverlap(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[8] = uint32_t(c.x[1]);
}

void hg_BlendStateAdvancedOverlap(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[8 + 0] : 0;
}

void hs_BlendStateAdvancedPremultipliedSrc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[9] = (c.x[1]) & 0xff;
}

void hg_BlendStateAdvancedPremultipliedSrc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[9 + 0] : 0;
}

void hs_BlendStateAdvancedNormalizedDst(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[10] = (c.x[1]) & 0xff;
}

void hg_BlendStateAdvancedNormalizedDst(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[10 + 0] : 0;
}

void hd_ColorState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_ColorStateBlendEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    const uint32_t key = uint32_t(c.x[1]);
    {
        const uint32_t k = key;
        d->f[0 + (k & 7) * 1 + 0] = (c.x[2]) & 0xff;
    }
}

void hg_ColorStateBlendEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    const uint32_t k = uint32_t(c.x[1]);
    c.x[0] = d ? d->f[0 + (k & 7) * 1 + 0] : 0;
}

void hs_ColorStateLogicOp(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[8] = uint32_t(c.x[1]);
}

void hg_ColorStateLogicOp(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[8 + 0] : 0;
}

void hs_ColorStateAlphaTest(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[9] = uint32_t(c.x[1]);
}

void hg_ColorStateAlphaTest(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[9 + 0] : 0;
}

void hd_ChannelMaskState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_ChannelMaskStateChannelMask(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    const uint32_t key = uint32_t(c.x[1]);
    {
        const uint32_t k = key;
        d->f[0 + (k & 7) * 4 + 0] = (c.x[2]) & 0xff;
        d->f[0 + (k & 7) * 4 + 1] = (c.x[3]) & 0xff;
        d->f[0 + (k & 7) * 4 + 2] = (c.x[4]) & 0xff;
        d->f[0 + (k & 7) * 4 + 3] = (c.x[5]) & 0xff;
    }
}

void hg_ChannelMaskStateChannelMask(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    const uint32_t k = uint32_t(c.x[1]);
    if (!d) return;
    put8(c.x[2], uint8_t(d->f[0 + (k & 7) * 4 + 0]));
    put8(c.x[3], uint8_t(d->f[0 + (k & 7) * 4 + 1]));
    put8(c.x[4], uint8_t(d->f[0 + (k & 7) * 4 + 2]));
    put8(c.x[5], uint8_t(d->f[0 + (k & 7) * 4 + 3]));
}

void hd_DepthStencilState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_DepthStencilStateDepthTestEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = (c.x[1]) & 0xff;
}

void hg_DepthStencilStateDepthTestEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[0 + 0] : 0;
}

void hs_DepthStencilStateDepthWriteEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[1] = (c.x[1]) & 0xff;
}

void hg_DepthStencilStateDepthWriteEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1 + 0] : 0;
}

void hs_DepthStencilStateDepthFunc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[2] = uint32_t(c.x[1]);
}

void hg_DepthStencilStateDepthFunc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2 + 0] : 0;
}

void hs_DepthStencilStateStencilTestEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[3] = (c.x[1]) & 0xff;
}

void hg_DepthStencilStateStencilTestEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[3 + 0] : 0;
}

void hs_DepthStencilStateStencilFunc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    const uint32_t key = uint32_t(c.x[1]);
    const uint32_t lo = key == 3 ? 1 : key, hi = key;  // both faces: front (1), back (2) and both (3)
    for (uint32_t k = lo; k <= hi; k++) {
        d->f[4 + (k & 7) * 1 + 0] = uint32_t(c.x[2]);
    }
}

void hg_DepthStencilStateStencilFunc(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    const uint32_t k = uint32_t(c.x[1]);
    c.x[0] = d ? d->f[4 + (k & 7) * 1 + 0] : 0;
}

void hs_DepthStencilStateStencilOp(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    const uint32_t key = uint32_t(c.x[1]);
    const uint32_t lo = key == 3 ? 1 : key, hi = key;  // both faces: front (1), back (2) and both (3)
    for (uint32_t k = lo; k <= hi; k++) {
        d->f[12 + (k & 7) * 3 + 0] = uint32_t(c.x[2]);
        d->f[12 + (k & 7) * 3 + 1] = uint32_t(c.x[3]);
        d->f[12 + (k & 7) * 3 + 2] = uint32_t(c.x[4]);
    }
}

void hg_DepthStencilStateStencilOp(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    const uint32_t k = uint32_t(c.x[1]);
    if (!d) return;
    put32(c.x[2], uint32_t(d->f[12 + (k & 7) * 3 + 0]));
    put32(c.x[3], uint32_t(d->f[12 + (k & 7) * 3 + 1]));
    put32(c.x[4], uint32_t(d->f[12 + (k & 7) * 3 + 2]));
}

void hd_MultisampleState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_MultisampleStateMultisampleEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = (c.x[1]) & 0xff;
}

void hg_MultisampleStateMultisampleEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[0 + 0] : 0;
}

void hs_MultisampleStateSamples(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[1] = uint32_t(c.x[1]);
}

void hg_MultisampleStateSamples(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1 + 0] : 0;
}

void hs_MultisampleStateAlphaToCoverageEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[2] = (c.x[1]) & 0xff;
}

void hg_MultisampleStateAlphaToCoverageEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2 + 0] : 0;
}

void hs_MultisampleStateAlphaToCoverageDither(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[3] = (c.x[1]) & 0xff;
}

void hg_MultisampleStateAlphaToCoverageDither(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[3 + 0] : 0;
}

void hs_MultisampleStateRasterSamples(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[4] = uint32_t(c.x[1]);
}

void hg_MultisampleStateRasterSamples(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[4 + 0] : 0;
}

void hs_MultisampleStateCoverageModulationMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[5] = uint32_t(c.x[1]);
}

void hg_MultisampleStateCoverageModulationMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[5 + 0] : 0;
}

void hs_MultisampleStateCoverageToColorEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[6] = (c.x[1]) & 0xff;
}

void hg_MultisampleStateCoverageToColorEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[6 + 0] : 0;
}

void hs_MultisampleStateCoverageToColorOutput(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[7] = uint32_t(c.x[1]);
}

void hg_MultisampleStateCoverageToColorOutput(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[7 + 0] : 0;
}

void hs_MultisampleStateSampleLocationsEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[8] = (c.x[1]) & 0xff;
}

void hg_MultisampleStateSampleLocationsEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[8 + 0] : 0;
}

void hs_MultisampleStateSampleLocationsGridEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[9] = (c.x[1]) & 0xff;
}

void hg_MultisampleStateSampleLocationsGridEnable(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[9 + 0] : 0;
}

void hs_MultisampleStateSampleLocations(CPU& c) {
    (void)c;  // custom sample positions are not kept: the replay uses the default pattern
}

void hg_MultisampleStateSampleLocationsGrid(CPU& c) {
    put32(c.x[1], 1);  // width
    put32(c.x[2], 1);  // height: no programmable grid
}

void hd_PolygonState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_PolygonStateCullFace(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = uint32_t(c.x[1]);
}

void hg_PolygonStateCullFace(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[0 + 0] : 0;
}

void hs_PolygonStateFrontFace(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[1] = uint32_t(c.x[1]);
}

void hg_PolygonStateFrontFace(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1 + 0] : 0;
}

void hs_PolygonStatePolygonMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[2] = uint32_t(c.x[1]);
}

void hg_PolygonStatePolygonMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2 + 0] : 0;
}

void hs_PolygonStatePolygonOffsetEnables(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[3] = uint32_t(c.x[1]);
}

void hg_PolygonStatePolygonOffsetEnables(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[3 + 0] : 0;
}

void hd_VertexAttribState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_VertexAttribStateFormat(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = uint32_t(c.x[1]);
    d->f[1] = c.x[2];
}

void hg_VertexAttribStateFormat(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    put32(c.x[1], uint32_t(d->f[0 + 0]));
    put64(c.x[2], d->f[0 + 1]);
}

void hs_VertexAttribStateStreamIndex(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[2] = uint32_t(c.x[1]);
}

void hg_VertexAttribStateStreamIndex(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2 + 0] : 0;
}

void hd_VertexStreamState(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_VertexStreamStateStride(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = c.x[1];
}

void hg_VertexStreamStateStride(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[0 + 0] : 0;
}

void hs_VertexStreamStateDivisor(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[1] = uint32_t(c.x[1]);
}

void hg_VertexStreamStateDivisor(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[1 + 0] : 0;
}

void hd_TextureView(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    if (BuilderData* d = bd(c.x[0])) std::memset(d, 0, sizeof *d);
}

void hs_TextureViewLevels(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[0] = uint32_t(c.x[1]);
    d->f[1] = uint32_t(c.x[2]);
    d->f[2] = 1;
}

void hg_TextureViewLevels(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !d->f[2]) {
        c.x[0] = 0;
        return;
    }
    put32(c.x[1], uint32_t(d->f[0 + 0]));
    put32(c.x[2], uint32_t(d->f[0 + 1]));
    c.x[0] = 1;
}

void hs_TextureViewLayers(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[3] = uint32_t(c.x[1]);
    d->f[4] = uint32_t(c.x[2]);
    d->f[5] = 1;
}

void hg_TextureViewLayers(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !d->f[5]) {
        c.x[0] = 0;
        return;
    }
    put32(c.x[1], uint32_t(d->f[3 + 0]));
    put32(c.x[2], uint32_t(d->f[3 + 1]));
    c.x[0] = 1;
}

void hs_TextureViewFormat(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[6] = uint32_t(c.x[1]);
    d->f[7] = 1;
}

void hg_TextureViewFormat(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !d->f[7]) {
        c.x[0] = 0;
        return;
    }
    put32(c.x[1], uint32_t(d->f[6 + 0]));
    c.x[0] = 1;
}

void hs_TextureViewSwizzle(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[8] = uint32_t(c.x[1]);
    d->f[9] = uint32_t(c.x[2]);
    d->f[10] = uint32_t(c.x[3]);
    d->f[11] = uint32_t(c.x[4]);
    d->f[12] = 1;
}

void hg_TextureViewSwizzle(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !d->f[12]) {
        c.x[0] = 0;
        return;
    }
    put32(c.x[1], uint32_t(d->f[8 + 0]));
    put32(c.x[2], uint32_t(d->f[8 + 1]));
    put32(c.x[3], uint32_t(d->f[8 + 2]));
    put32(c.x[4], uint32_t(d->f[8 + 3]));
    c.x[0] = 1;
}

void hs_TextureViewDepthStencilMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[13] = uint32_t(c.x[1]);
    d->f[14] = 1;
}

void hg_TextureViewDepthStencilMode(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !d->f[14]) {
        c.x[0] = 0;
        return;
    }
    put32(c.x[1], uint32_t(d->f[13 + 0]));
    c.x[0] = 1;
}

void hs_TextureViewTarget(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d) return;
    d->f[15] = uint32_t(c.x[1]);
    d->f[16] = 1;
}

void hg_TextureViewTarget(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    if (!d || !d->f[16]) {
        c.x[0] = 0;
        return;
    }
    put32(c.x[1], uint32_t(d->f[15 + 0]));
    c.x[0] = 1;
}

void hc_TextureView(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* a = bd(c.x[0]);
    BuilderData* b = bd(c.x[1]);
    c.x[0] = a && b && std::memcmp(a->f, b->f, 17 * sizeof(uint64_t)) == 0 ? 1 : 0;
}

void hb_WindowBuilderGetNumTextures(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[3] : 0;
}

void hb_WindowBuilderGetTexture(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    const uint64_t index = uint32_t(c.x[1]);
    c.x[0] = d && d->f[2] && index < d->f[3] ? reinterpret_cast<const uint64_t*>(d->f[2])[index] : 0;
}

void hb_WindowBuilderGetTextures(CPU& c) {
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[2] : 0;
}

void hb_TextureBuilderGetPackagedTextureLayout(CPU& c) {
    // 0 when no layout was set (a game that does not use packaged textures); else the pointer the game gave.
    std::lock_guard<std::mutex> lk(g_bd_lock);
    BuilderData* d = bd(c.x[0]);
    c.x[0] = d ? d->f[16] : 0;
}

using HostFn = void (*)(CPU&);
struct HostProc {
    const char* name;
    HostFn fn;
};

// Every proc the host implements for the builders, by NVN name (each is also registered as "host_<name>" below).
const HostProc kHostBuilderProcs[] = {
    {"nvnMemoryPoolBuilderSetDefaults", host_pool_builder_set_defaults},
    {"nvnMemoryPoolBuilderSetDevice", host_pool_builder_set_device},
    {"nvnMemoryPoolBuilderSetFlags", host_pool_builder_set_flags},
    {"nvnMemoryPoolBuilderSetStorage", host_pool_builder_set_storage},
    {"nvnMemoryPoolBuilderGetDevice", host_pool_builder_get_device},
    {"nvnMemoryPoolBuilderGetMemory", host_pool_builder_get_memory},
    {"nvnMemoryPoolBuilderGetSize", host_pool_builder_get_size},
    {"nvnMemoryPoolBuilderGetFlags", host_pool_builder_get_flags},
    {"nvnDeviceBuilderSetDefaults", hb_DeviceBuilderSetDefaults},
    {"nvnDeviceBuilderSetFlags", hb_DeviceBuilderSetFlags},
    {"nvnQueueBuilderSetDefaults", hb_QueueBuilderSetDefaults},
    {"nvnQueueBuilderSetDevice", hb_QueueBuilderSetDevice},
    {"nvnQueueBuilderSetFlags", hb_QueueBuilderSetFlags},
    {"nvnQueueBuilderSetCommandMemorySize", hb_QueueBuilderSetCommandMemorySize},
    {"nvnQueueBuilderSetComputeMemorySize", hb_QueueBuilderSetComputeMemorySize},
    {"nvnQueueBuilderSetControlMemorySize", hb_QueueBuilderSetControlMemorySize},
    {"nvnQueueBuilderSetCommandFlushThreshold", hb_QueueBuilderSetCommandFlushThreshold},
    {"nvnQueueBuilderSetQueueMemory", hb_QueueBuilderSetQueueMemory},
    {"nvnQueueBuilderGetQueueMemorySize", hb_QueueBuilderGetQueueMemorySize},
    {"nvnWindowBuilderSetDefaults", hb_WindowBuilderSetDefaults},
    {"nvnWindowBuilderSetDevice", hb_WindowBuilderSetDevice},
    {"nvnWindowBuilderSetNativeWindow", hb_WindowBuilderSetNativeWindow},
    {"nvnWindowBuilderSetTextures", hb_WindowBuilderSetTextures},
    {"nvnWindowBuilderSetPresentInterval", hb_WindowBuilderSetPresentInterval},
    {"nvnWindowBuilderGetNativeWindow", hb_WindowBuilderGetNativeWindow},
    {"nvnWindowBuilderGetPresentInterval", hb_WindowBuilderGetPresentInterval},
    {"nvnBufferBuilderSetDefaults", hb_BufferBuilderSetDefaults},
    {"nvnBufferBuilderSetDevice", hb_BufferBuilderSetDevice},
    {"nvnBufferBuilderSetStorage", hb_BufferBuilderSetStorage},
    {"nvnBufferBuilderGetMemoryPool", hb_BufferBuilderGetMemoryPool},
    {"nvnBufferBuilderGetMemoryOffset", hb_BufferBuilderGetMemoryOffset},
    {"nvnBufferBuilderGetSize", hb_BufferBuilderGetSize},
    {"nvnSamplerBuilderSetDefaults", hb_SamplerBuilderSetDefaults},
    {"nvnSamplerBuilderSetDevice", hb_SamplerBuilderSetDevice},
    {"nvnSamplerBuilderSetMinMagFilter", hb_SamplerBuilderSetMinMagFilter},
    {"nvnSamplerBuilderSetWrapMode", hb_SamplerBuilderSetWrapMode},
    {"nvnSamplerBuilderSetLodClamp", hb_SamplerBuilderSetLodClamp},
    {"nvnSamplerBuilderSetLodBias", hb_SamplerBuilderSetLodBias},
    {"nvnSamplerBuilderSetCompare", hb_SamplerBuilderSetCompare},
    {"nvnSamplerBuilderSetBorderColor", hb_SamplerBuilderSetBorderColor},
    {"nvnSamplerBuilderGetBorderColor", hb_SamplerBuilderGetBorderColor},
    {"nvnSamplerBuilderSetBorderColori", hb_SamplerBuilderSetBorderColori},
    {"nvnSamplerBuilderGetBorderColori", hb_SamplerBuilderGetBorderColori},
    {"nvnSamplerBuilderSetBorderColorui", hb_SamplerBuilderSetBorderColorui},
    {"nvnSamplerBuilderGetBorderColorui", hb_SamplerBuilderGetBorderColorui},
    {"nvnSamplerBuilderSetMaxAnisotropy", hb_SamplerBuilderSetMaxAnisotropy},
    {"nvnSamplerBuilderSetReductionFilter", hb_SamplerBuilderSetReductionFilter},
    {"nvnSamplerBuilderGetMinMagFilter", hb_SamplerBuilderGetMinMagFilter},
    {"nvnSamplerBuilderGetWrapMode", hb_SamplerBuilderGetWrapMode},
    {"nvnSamplerBuilderGetLodClamp", hb_SamplerBuilderGetLodClamp},
    {"nvnSamplerBuilderGetLodBias", hb_SamplerBuilderGetLodBias},
    {"nvnSamplerBuilderGetCompare", hb_SamplerBuilderGetCompare},
    {"nvnSamplerBuilderGetMaxAnisotropy", hb_SamplerBuilderGetMaxAnisotropy},
    {"nvnSamplerBuilderGetReductionFilter", hb_SamplerBuilderGetReductionFilter},
    {"nvnTextureBuilderSetDefaults", hb_TextureBuilderSetDefaults},
    {"nvnTextureBuilderSetDevice", hb_TextureBuilderSetDevice},
    {"nvnTextureBuilderSetFlags", hb_TextureBuilderSetFlags},
    {"nvnTextureBuilderSetTarget", hb_TextureBuilderSetTarget},
    {"nvnTextureBuilderSetWidth", hb_TextureBuilderSetWidth},
    {"nvnTextureBuilderSetHeight", hb_TextureBuilderSetHeight},
    {"nvnTextureBuilderSetDepth", hb_TextureBuilderSetDepth},
    {"nvnTextureBuilderSetSize1D", hb_TextureBuilderSetSize1D},
    {"nvnTextureBuilderSetSize2D", hb_TextureBuilderSetSize2D},
    {"nvnTextureBuilderSetSize3D", hb_TextureBuilderSetSize3D},
    {"nvnTextureBuilderSetLevels", hb_TextureBuilderSetLevels},
    {"nvnTextureBuilderSetFormat", hb_TextureBuilderSetFormat},
    {"nvnTextureBuilderSetSamples", hb_TextureBuilderSetSamples},
    {"nvnTextureBuilderSetSwizzle", hb_TextureBuilderSetSwizzle},
    {"nvnTextureBuilderSetDepthStencilMode", hb_TextureBuilderSetDepthStencilMode},
    {"nvnTextureBuilderSetStorage", hb_TextureBuilderSetStorage},
    {"nvnTextureBuilderSetStride", hb_TextureBuilderSetStride},
    {"nvnTextureBuilderSetGLTextureName", hb_TextureBuilderSetGLTextureName},
    {"nvnTextureBuilderSetPackagedTextureData", hb_TextureBuilderSetPackagedTextureData},
    {"nvnTextureBuilderSetPackagedTextureLayout", hb_TextureBuilderSetPackagedTextureLayout},
    {"nvnTextureBuilderGetFlags", hb_TextureBuilderGetFlags},
    {"nvnTextureBuilderGetTarget", hb_TextureBuilderGetTarget},
    {"nvnTextureBuilderGetWidth", hb_TextureBuilderGetWidth},
    {"nvnTextureBuilderGetHeight", hb_TextureBuilderGetHeight},
    {"nvnTextureBuilderGetDepth", hb_TextureBuilderGetDepth},
    {"nvnTextureBuilderGetLevels", hb_TextureBuilderGetLevels},
    {"nvnTextureBuilderGetFormat", hb_TextureBuilderGetFormat},
    {"nvnTextureBuilderGetSamples", hb_TextureBuilderGetSamples},
    {"nvnTextureBuilderGetDepthStencilMode", hb_TextureBuilderGetDepthStencilMode},
    {"nvnTextureBuilderGetStride", hb_TextureBuilderGetStride},
    {"nvnTextureBuilderGetGLTextureName", hb_TextureBuilderGetGLTextureName},
    {"nvnTextureBuilderGetMemoryPool", hb_TextureBuilderGetMemoryPool},
    {"nvnTextureBuilderGetMemoryOffset", hb_TextureBuilderGetMemoryOffset},
    {"nvnTextureBuilderGetPackagedTextureData", hb_TextureBuilderGetPackagedTextureData},
    {"nvnTextureBuilderGetSwizzle", hb_TextureBuilderGetSwizzle},
    {"nvnBlendStateSetDefaults", hd_BlendState},
    {"nvnBlendStateSetBlendTarget", hs_BlendStateBlendTarget},
    {"nvnBlendStateGetBlendTarget", hg_BlendStateBlendTarget},
    {"nvnBlendStateSetBlendFunc", hs_BlendStateBlendFunc},
    {"nvnBlendStateGetBlendFunc", hg_BlendStateBlendFunc},
    {"nvnBlendStateSetBlendEquation", hs_BlendStateBlendEquation},
    {"nvnBlendStateGetBlendEquation", hg_BlendStateBlendEquation},
    {"nvnBlendStateSetAdvancedMode", hs_BlendStateAdvancedMode},
    {"nvnBlendStateGetAdvancedMode", hg_BlendStateAdvancedMode},
    {"nvnBlendStateSetAdvancedOverlap", hs_BlendStateAdvancedOverlap},
    {"nvnBlendStateGetAdvancedOverlap", hg_BlendStateAdvancedOverlap},
    {"nvnBlendStateSetAdvancedPremultipliedSrc", hs_BlendStateAdvancedPremultipliedSrc},
    {"nvnBlendStateGetAdvancedPremultipliedSrc", hg_BlendStateAdvancedPremultipliedSrc},
    {"nvnBlendStateSetAdvancedNormalizedDst", hs_BlendStateAdvancedNormalizedDst},
    {"nvnBlendStateGetAdvancedNormalizedDst", hg_BlendStateAdvancedNormalizedDst},
    {"nvnColorStateSetDefaults", hd_ColorState},
    {"nvnColorStateSetBlendEnable", hs_ColorStateBlendEnable},
    {"nvnColorStateGetBlendEnable", hg_ColorStateBlendEnable},
    {"nvnColorStateSetLogicOp", hs_ColorStateLogicOp},
    {"nvnColorStateGetLogicOp", hg_ColorStateLogicOp},
    {"nvnColorStateSetAlphaTest", hs_ColorStateAlphaTest},
    {"nvnColorStateGetAlphaTest", hg_ColorStateAlphaTest},
    {"nvnChannelMaskStateSetDefaults", hd_ChannelMaskState},
    {"nvnChannelMaskStateSetChannelMask", hs_ChannelMaskStateChannelMask},
    {"nvnChannelMaskStateGetChannelMask", hg_ChannelMaskStateChannelMask},
    {"nvnDepthStencilStateSetDefaults", hd_DepthStencilState},
    {"nvnDepthStencilStateSetDepthTestEnable", hs_DepthStencilStateDepthTestEnable},
    {"nvnDepthStencilStateGetDepthTestEnable", hg_DepthStencilStateDepthTestEnable},
    {"nvnDepthStencilStateSetDepthWriteEnable", hs_DepthStencilStateDepthWriteEnable},
    {"nvnDepthStencilStateGetDepthWriteEnable", hg_DepthStencilStateDepthWriteEnable},
    {"nvnDepthStencilStateSetDepthFunc", hs_DepthStencilStateDepthFunc},
    {"nvnDepthStencilStateGetDepthFunc", hg_DepthStencilStateDepthFunc},
    {"nvnDepthStencilStateSetStencilTestEnable", hs_DepthStencilStateStencilTestEnable},
    {"nvnDepthStencilStateGetStencilTestEnable", hg_DepthStencilStateStencilTestEnable},
    {"nvnDepthStencilStateSetStencilFunc", hs_DepthStencilStateStencilFunc},
    {"nvnDepthStencilStateGetStencilFunc", hg_DepthStencilStateStencilFunc},
    {"nvnDepthStencilStateSetStencilOp", hs_DepthStencilStateStencilOp},
    {"nvnDepthStencilStateGetStencilOp", hg_DepthStencilStateStencilOp},
    {"nvnMultisampleStateSetDefaults", hd_MultisampleState},
    {"nvnMultisampleStateSetMultisampleEnable", hs_MultisampleStateMultisampleEnable},
    {"nvnMultisampleStateGetMultisampleEnable", hg_MultisampleStateMultisampleEnable},
    {"nvnMultisampleStateSetSamples", hs_MultisampleStateSamples},
    {"nvnMultisampleStateGetSamples", hg_MultisampleStateSamples},
    {"nvnMultisampleStateSetAlphaToCoverageEnable", hs_MultisampleStateAlphaToCoverageEnable},
    {"nvnMultisampleStateGetAlphaToCoverageEnable", hg_MultisampleStateAlphaToCoverageEnable},
    {"nvnMultisampleStateSetAlphaToCoverageDither", hs_MultisampleStateAlphaToCoverageDither},
    {"nvnMultisampleStateGetAlphaToCoverageDither", hg_MultisampleStateAlphaToCoverageDither},
    {"nvnMultisampleStateSetRasterSamples", hs_MultisampleStateRasterSamples},
    {"nvnMultisampleStateGetRasterSamples", hg_MultisampleStateRasterSamples},
    {"nvnMultisampleStateSetCoverageModulationMode", hs_MultisampleStateCoverageModulationMode},
    {"nvnMultisampleStateGetCoverageModulationMode", hg_MultisampleStateCoverageModulationMode},
    {"nvnMultisampleStateSetCoverageToColorEnable", hs_MultisampleStateCoverageToColorEnable},
    {"nvnMultisampleStateGetCoverageToColorEnable", hg_MultisampleStateCoverageToColorEnable},
    {"nvnMultisampleStateSetCoverageToColorOutput", hs_MultisampleStateCoverageToColorOutput},
    {"nvnMultisampleStateGetCoverageToColorOutput", hg_MultisampleStateCoverageToColorOutput},
    {"nvnMultisampleStateSetSampleLocationsEnable", hs_MultisampleStateSampleLocationsEnable},
    {"nvnMultisampleStateGetSampleLocationsEnable", hg_MultisampleStateSampleLocationsEnable},
    {"nvnMultisampleStateSetSampleLocationsGridEnable", hs_MultisampleStateSampleLocationsGridEnable},
    {"nvnMultisampleStateGetSampleLocationsGridEnable", hg_MultisampleStateSampleLocationsGridEnable},
    {"nvnMultisampleStateSetSampleLocations", hs_MultisampleStateSampleLocations},
    {"nvnMultisampleStateGetSampleLocationsGrid", hg_MultisampleStateSampleLocationsGrid},
    {"nvnPolygonStateSetDefaults", hd_PolygonState},
    {"nvnPolygonStateSetCullFace", hs_PolygonStateCullFace},
    {"nvnPolygonStateGetCullFace", hg_PolygonStateCullFace},
    {"nvnPolygonStateSetFrontFace", hs_PolygonStateFrontFace},
    {"nvnPolygonStateGetFrontFace", hg_PolygonStateFrontFace},
    {"nvnPolygonStateSetPolygonMode", hs_PolygonStatePolygonMode},
    {"nvnPolygonStateGetPolygonMode", hg_PolygonStatePolygonMode},
    {"nvnPolygonStateSetPolygonOffsetEnables", hs_PolygonStatePolygonOffsetEnables},
    {"nvnPolygonStateGetPolygonOffsetEnables", hg_PolygonStatePolygonOffsetEnables},
    {"nvnVertexAttribStateSetDefaults", hd_VertexAttribState},
    {"nvnVertexAttribStateSetFormat", hs_VertexAttribStateFormat},
    {"nvnVertexAttribStateGetFormat", hg_VertexAttribStateFormat},
    {"nvnVertexAttribStateSetStreamIndex", hs_VertexAttribStateStreamIndex},
    {"nvnVertexAttribStateGetStreamIndex", hg_VertexAttribStateStreamIndex},
    {"nvnVertexStreamStateSetDefaults", hd_VertexStreamState},
    {"nvnVertexStreamStateSetStride", hs_VertexStreamStateStride},
    {"nvnVertexStreamStateGetStride", hg_VertexStreamStateStride},
    {"nvnVertexStreamStateSetDivisor", hs_VertexStreamStateDivisor},
    {"nvnVertexStreamStateGetDivisor", hg_VertexStreamStateDivisor},
    {"nvnTextureViewSetDefaults", hd_TextureView},
    {"nvnTextureViewSetLevels", hs_TextureViewLevels},
    {"nvnTextureViewGetLevels", hg_TextureViewLevels},
    {"nvnTextureViewSetLayers", hs_TextureViewLayers},
    {"nvnTextureViewGetLayers", hg_TextureViewLayers},
    {"nvnTextureViewSetFormat", hs_TextureViewFormat},
    {"nvnTextureViewGetFormat", hg_TextureViewFormat},
    {"nvnTextureViewSetSwizzle", hs_TextureViewSwizzle},
    {"nvnTextureViewGetSwizzle", hg_TextureViewSwizzle},
    {"nvnTextureViewSetDepthStencilMode", hs_TextureViewDepthStencilMode},
    {"nvnTextureViewGetDepthStencilMode", hg_TextureViewDepthStencilMode},
    {"nvnTextureViewSetTarget", hs_TextureViewTarget},
    {"nvnTextureViewGetTarget", hg_TextureViewTarget},
    {"nvnTextureViewCompare", hc_TextureView},
    {"nvnWindowBuilderGetNumTextures", hb_WindowBuilderGetNumTextures},
    {"nvnWindowBuilderGetTexture", hb_WindowBuilderGetTexture},
    {"nvnWindowBuilderGetTextures", hb_WindowBuilderGetTextures},
    {"nvnTextureBuilderGetPackagedTextureLayout", hb_TextureBuilderGetPackagedTextureLayout},
};

std::mutex g_host_lock;
std::unordered_map<uint64_t, HostFn> g_host_by_address;  // the HLE address handed out -> the function behind it

const HostProc* find_host_proc(const char* name) {
    static const std::unordered_map<std::string, const HostProc*> by_name = [] {
        std::unordered_map<std::string, const HostProc*> m;
        for (const HostProc& p : kHostBuilderProcs) m[p.name] = &p;
        return m;
    }();
    auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : it->second;
}

// The host's version of a builder proc as an HLE address (what the game, or a runtime caller, can call); 0 if none.
uint64_t host_proc_address(const char* name) {
    const HostProc* p = find_host_proc(name);
    if (!p) return 0;
    const std::string host = std::string("host_") + name;
    if (!hle_lookup(host.c_str())) return 0;
    const uint64_t addr = hle_address(host.c_str());
    std::lock_guard lk(g_host_lock);
    g_host_by_address[addr] = p->fn;
    return addr;
}

HostFn host_fn_for(uint64_t addr) {
    std::lock_guard lk(g_host_lock);
    auto it = g_host_by_address.find(addr);
    return it == g_host_by_address.end() ? nullptr : it->second;
}

uint64_t host_builder_proc(const std::string& name) { return host_proc_address(name.c_str()); }

// What a runtime caller of nvn::real_proc gets for a proc that no driver and no host version serves: a function that
// stops with the proc's name, instead of a null call that faults at pc 0 with nothing to say which proc it was.
void missing_driver_proc(CPU& c) {
    fatal("nvn: %s: this game's sdk has no NVN driver and the host has no version of it yet (lr %s)",
          hle_current_name(), describe(c.x[30]).c_str());
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
    if (note_all) nvn::driver_proc(c, name.c_str());
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
            if (uint64_t real = nvn::driver_proc(c, name.c_str())) {
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
NX_HLE("host_nvnDeviceBuilderSetDefaults", hb_DeviceBuilderSetDefaults);
NX_HLE("host_nvnDeviceBuilderSetFlags", hb_DeviceBuilderSetFlags);
NX_HLE("host_nvnQueueBuilderSetDefaults", hb_QueueBuilderSetDefaults);
NX_HLE("host_nvnQueueBuilderSetDevice", hb_QueueBuilderSetDevice);
NX_HLE("host_nvnQueueBuilderSetFlags", hb_QueueBuilderSetFlags);
NX_HLE("host_nvnQueueBuilderSetCommandMemorySize", hb_QueueBuilderSetCommandMemorySize);
NX_HLE("host_nvnQueueBuilderSetComputeMemorySize", hb_QueueBuilderSetComputeMemorySize);
NX_HLE("host_nvnQueueBuilderSetControlMemorySize", hb_QueueBuilderSetControlMemorySize);
NX_HLE("host_nvnQueueBuilderSetCommandFlushThreshold", hb_QueueBuilderSetCommandFlushThreshold);
NX_HLE("host_nvnQueueBuilderSetQueueMemory", hb_QueueBuilderSetQueueMemory);
NX_HLE("host_nvnQueueBuilderGetQueueMemorySize", hb_QueueBuilderGetQueueMemorySize);
NX_HLE("host_nvnWindowBuilderSetDefaults", hb_WindowBuilderSetDefaults);
NX_HLE("host_nvnWindowBuilderSetDevice", hb_WindowBuilderSetDevice);
NX_HLE("host_nvnWindowBuilderSetNativeWindow", hb_WindowBuilderSetNativeWindow);
NX_HLE("host_nvnWindowBuilderSetTextures", hb_WindowBuilderSetTextures);
NX_HLE("host_nvnWindowBuilderSetPresentInterval", hb_WindowBuilderSetPresentInterval);
NX_HLE("host_nvnWindowBuilderGetNativeWindow", hb_WindowBuilderGetNativeWindow);
NX_HLE("host_nvnWindowBuilderGetPresentInterval", hb_WindowBuilderGetPresentInterval);
NX_HLE("host_nvnBufferBuilderSetDefaults", hb_BufferBuilderSetDefaults);
NX_HLE("host_nvnBufferBuilderSetDevice", hb_BufferBuilderSetDevice);
NX_HLE("host_nvnBufferBuilderSetStorage", hb_BufferBuilderSetStorage);
NX_HLE("host_nvnBufferBuilderGetMemoryPool", hb_BufferBuilderGetMemoryPool);
NX_HLE("host_nvnBufferBuilderGetMemoryOffset", hb_BufferBuilderGetMemoryOffset);
NX_HLE("host_nvnBufferBuilderGetSize", hb_BufferBuilderGetSize);
NX_HLE("host_nvnSamplerBuilderSetDefaults", hb_SamplerBuilderSetDefaults);
NX_HLE("host_nvnSamplerBuilderSetDevice", hb_SamplerBuilderSetDevice);
NX_HLE("host_nvnSamplerBuilderSetMinMagFilter", hb_SamplerBuilderSetMinMagFilter);
NX_HLE("host_nvnSamplerBuilderSetWrapMode", hb_SamplerBuilderSetWrapMode);
NX_HLE("host_nvnSamplerBuilderSetLodClamp", hb_SamplerBuilderSetLodClamp);
NX_HLE("host_nvnSamplerBuilderSetLodBias", hb_SamplerBuilderSetLodBias);
NX_HLE("host_nvnSamplerBuilderSetCompare", hb_SamplerBuilderSetCompare);
NX_HLE("host_nvnSamplerBuilderSetBorderColor", hb_SamplerBuilderSetBorderColor);
NX_HLE("host_nvnSamplerBuilderGetBorderColor", hb_SamplerBuilderGetBorderColor);
NX_HLE("host_nvnSamplerBuilderSetBorderColori", hb_SamplerBuilderSetBorderColori);
NX_HLE("host_nvnSamplerBuilderGetBorderColori", hb_SamplerBuilderGetBorderColori);
NX_HLE("host_nvnSamplerBuilderSetBorderColorui", hb_SamplerBuilderSetBorderColorui);
NX_HLE("host_nvnSamplerBuilderGetBorderColorui", hb_SamplerBuilderGetBorderColorui);
NX_HLE("host_nvnSamplerBuilderSetMaxAnisotropy", hb_SamplerBuilderSetMaxAnisotropy);
NX_HLE("host_nvnSamplerBuilderSetReductionFilter", hb_SamplerBuilderSetReductionFilter);
NX_HLE("host_nvnSamplerBuilderGetMinMagFilter", hb_SamplerBuilderGetMinMagFilter);
NX_HLE("host_nvnSamplerBuilderGetWrapMode", hb_SamplerBuilderGetWrapMode);
NX_HLE("host_nvnSamplerBuilderGetLodClamp", hb_SamplerBuilderGetLodClamp);
NX_HLE("host_nvnSamplerBuilderGetLodBias", hb_SamplerBuilderGetLodBias);
NX_HLE("host_nvnSamplerBuilderGetCompare", hb_SamplerBuilderGetCompare);
NX_HLE("host_nvnSamplerBuilderGetMaxAnisotropy", hb_SamplerBuilderGetMaxAnisotropy);
NX_HLE("host_nvnSamplerBuilderGetReductionFilter", hb_SamplerBuilderGetReductionFilter);
NX_HLE("host_nvnTextureBuilderSetDefaults", hb_TextureBuilderSetDefaults);
NX_HLE("host_nvnTextureBuilderSetDevice", hb_TextureBuilderSetDevice);
NX_HLE("host_nvnTextureBuilderSetFlags", hb_TextureBuilderSetFlags);
NX_HLE("host_nvnTextureBuilderSetTarget", hb_TextureBuilderSetTarget);
NX_HLE("host_nvnTextureBuilderSetWidth", hb_TextureBuilderSetWidth);
NX_HLE("host_nvnTextureBuilderSetHeight", hb_TextureBuilderSetHeight);
NX_HLE("host_nvnTextureBuilderSetDepth", hb_TextureBuilderSetDepth);
NX_HLE("host_nvnTextureBuilderSetSize1D", hb_TextureBuilderSetSize1D);
NX_HLE("host_nvnTextureBuilderSetSize2D", hb_TextureBuilderSetSize2D);
NX_HLE("host_nvnTextureBuilderSetSize3D", hb_TextureBuilderSetSize3D);
NX_HLE("host_nvnTextureBuilderSetLevels", hb_TextureBuilderSetLevels);
NX_HLE("host_nvnTextureBuilderSetFormat", hb_TextureBuilderSetFormat);
NX_HLE("host_nvnTextureBuilderSetSamples", hb_TextureBuilderSetSamples);
NX_HLE("host_nvnTextureBuilderSetSwizzle", hb_TextureBuilderSetSwizzle);
NX_HLE("host_nvnTextureBuilderSetDepthStencilMode", hb_TextureBuilderSetDepthStencilMode);
NX_HLE("host_nvnTextureBuilderSetStorage", hb_TextureBuilderSetStorage);
NX_HLE("host_nvnTextureBuilderSetStride", hb_TextureBuilderSetStride);
NX_HLE("host_nvnTextureBuilderSetGLTextureName", hb_TextureBuilderSetGLTextureName);
NX_HLE("host_nvnTextureBuilderSetPackagedTextureData", hb_TextureBuilderSetPackagedTextureData);
NX_HLE("host_nvnTextureBuilderSetPackagedTextureLayout", hb_TextureBuilderSetPackagedTextureLayout);
NX_HLE("host_nvnTextureBuilderGetFlags", hb_TextureBuilderGetFlags);
NX_HLE("host_nvnTextureBuilderGetTarget", hb_TextureBuilderGetTarget);
NX_HLE("host_nvnTextureBuilderGetWidth", hb_TextureBuilderGetWidth);
NX_HLE("host_nvnTextureBuilderGetHeight", hb_TextureBuilderGetHeight);
NX_HLE("host_nvnTextureBuilderGetDepth", hb_TextureBuilderGetDepth);
NX_HLE("host_nvnTextureBuilderGetLevels", hb_TextureBuilderGetLevels);
NX_HLE("host_nvnTextureBuilderGetFormat", hb_TextureBuilderGetFormat);
NX_HLE("host_nvnTextureBuilderGetSamples", hb_TextureBuilderGetSamples);
NX_HLE("host_nvnTextureBuilderGetDepthStencilMode", hb_TextureBuilderGetDepthStencilMode);
NX_HLE("host_nvnTextureBuilderGetStride", hb_TextureBuilderGetStride);
NX_HLE("host_nvnTextureBuilderGetGLTextureName", hb_TextureBuilderGetGLTextureName);
NX_HLE("host_nvnTextureBuilderGetMemoryPool", hb_TextureBuilderGetMemoryPool);
NX_HLE("host_nvnTextureBuilderGetMemoryOffset", hb_TextureBuilderGetMemoryOffset);
NX_HLE("host_nvnTextureBuilderGetPackagedTextureData", hb_TextureBuilderGetPackagedTextureData);
NX_HLE("host_nvnTextureBuilderGetSwizzle", hb_TextureBuilderGetSwizzle);
NX_HLE("host_nvnBlendStateSetDefaults", hd_BlendState);
NX_HLE("host_nvnBlendStateSetBlendTarget", hs_BlendStateBlendTarget);
NX_HLE("host_nvnBlendStateGetBlendTarget", hg_BlendStateBlendTarget);
NX_HLE("host_nvnBlendStateSetBlendFunc", hs_BlendStateBlendFunc);
NX_HLE("host_nvnBlendStateGetBlendFunc", hg_BlendStateBlendFunc);
NX_HLE("host_nvnBlendStateSetBlendEquation", hs_BlendStateBlendEquation);
NX_HLE("host_nvnBlendStateGetBlendEquation", hg_BlendStateBlendEquation);
NX_HLE("host_nvnBlendStateSetAdvancedMode", hs_BlendStateAdvancedMode);
NX_HLE("host_nvnBlendStateGetAdvancedMode", hg_BlendStateAdvancedMode);
NX_HLE("host_nvnBlendStateSetAdvancedOverlap", hs_BlendStateAdvancedOverlap);
NX_HLE("host_nvnBlendStateGetAdvancedOverlap", hg_BlendStateAdvancedOverlap);
NX_HLE("host_nvnBlendStateSetAdvancedPremultipliedSrc", hs_BlendStateAdvancedPremultipliedSrc);
NX_HLE("host_nvnBlendStateGetAdvancedPremultipliedSrc", hg_BlendStateAdvancedPremultipliedSrc);
NX_HLE("host_nvnBlendStateSetAdvancedNormalizedDst", hs_BlendStateAdvancedNormalizedDst);
NX_HLE("host_nvnBlendStateGetAdvancedNormalizedDst", hg_BlendStateAdvancedNormalizedDst);
NX_HLE("host_nvnColorStateSetDefaults", hd_ColorState);
NX_HLE("host_nvnColorStateSetBlendEnable", hs_ColorStateBlendEnable);
NX_HLE("host_nvnColorStateGetBlendEnable", hg_ColorStateBlendEnable);
NX_HLE("host_nvnColorStateSetLogicOp", hs_ColorStateLogicOp);
NX_HLE("host_nvnColorStateGetLogicOp", hg_ColorStateLogicOp);
NX_HLE("host_nvnColorStateSetAlphaTest", hs_ColorStateAlphaTest);
NX_HLE("host_nvnColorStateGetAlphaTest", hg_ColorStateAlphaTest);
NX_HLE("host_nvnChannelMaskStateSetDefaults", hd_ChannelMaskState);
NX_HLE("host_nvnChannelMaskStateSetChannelMask", hs_ChannelMaskStateChannelMask);
NX_HLE("host_nvnChannelMaskStateGetChannelMask", hg_ChannelMaskStateChannelMask);
NX_HLE("host_nvnDepthStencilStateSetDefaults", hd_DepthStencilState);
NX_HLE("host_nvnDepthStencilStateSetDepthTestEnable", hs_DepthStencilStateDepthTestEnable);
NX_HLE("host_nvnDepthStencilStateGetDepthTestEnable", hg_DepthStencilStateDepthTestEnable);
NX_HLE("host_nvnDepthStencilStateSetDepthWriteEnable", hs_DepthStencilStateDepthWriteEnable);
NX_HLE("host_nvnDepthStencilStateGetDepthWriteEnable", hg_DepthStencilStateDepthWriteEnable);
NX_HLE("host_nvnDepthStencilStateSetDepthFunc", hs_DepthStencilStateDepthFunc);
NX_HLE("host_nvnDepthStencilStateGetDepthFunc", hg_DepthStencilStateDepthFunc);
NX_HLE("host_nvnDepthStencilStateSetStencilTestEnable", hs_DepthStencilStateStencilTestEnable);
NX_HLE("host_nvnDepthStencilStateGetStencilTestEnable", hg_DepthStencilStateStencilTestEnable);
NX_HLE("host_nvnDepthStencilStateSetStencilFunc", hs_DepthStencilStateStencilFunc);
NX_HLE("host_nvnDepthStencilStateGetStencilFunc", hg_DepthStencilStateStencilFunc);
NX_HLE("host_nvnDepthStencilStateSetStencilOp", hs_DepthStencilStateStencilOp);
NX_HLE("host_nvnDepthStencilStateGetStencilOp", hg_DepthStencilStateStencilOp);
NX_HLE("host_nvnMultisampleStateSetDefaults", hd_MultisampleState);
NX_HLE("host_nvnMultisampleStateSetMultisampleEnable", hs_MultisampleStateMultisampleEnable);
NX_HLE("host_nvnMultisampleStateGetMultisampleEnable", hg_MultisampleStateMultisampleEnable);
NX_HLE("host_nvnMultisampleStateSetSamples", hs_MultisampleStateSamples);
NX_HLE("host_nvnMultisampleStateGetSamples", hg_MultisampleStateSamples);
NX_HLE("host_nvnMultisampleStateSetAlphaToCoverageEnable", hs_MultisampleStateAlphaToCoverageEnable);
NX_HLE("host_nvnMultisampleStateGetAlphaToCoverageEnable", hg_MultisampleStateAlphaToCoverageEnable);
NX_HLE("host_nvnMultisampleStateSetAlphaToCoverageDither", hs_MultisampleStateAlphaToCoverageDither);
NX_HLE("host_nvnMultisampleStateGetAlphaToCoverageDither", hg_MultisampleStateAlphaToCoverageDither);
NX_HLE("host_nvnMultisampleStateSetRasterSamples", hs_MultisampleStateRasterSamples);
NX_HLE("host_nvnMultisampleStateGetRasterSamples", hg_MultisampleStateRasterSamples);
NX_HLE("host_nvnMultisampleStateSetCoverageModulationMode", hs_MultisampleStateCoverageModulationMode);
NX_HLE("host_nvnMultisampleStateGetCoverageModulationMode", hg_MultisampleStateCoverageModulationMode);
NX_HLE("host_nvnMultisampleStateSetCoverageToColorEnable", hs_MultisampleStateCoverageToColorEnable);
NX_HLE("host_nvnMultisampleStateGetCoverageToColorEnable", hg_MultisampleStateCoverageToColorEnable);
NX_HLE("host_nvnMultisampleStateSetCoverageToColorOutput", hs_MultisampleStateCoverageToColorOutput);
NX_HLE("host_nvnMultisampleStateGetCoverageToColorOutput", hg_MultisampleStateCoverageToColorOutput);
NX_HLE("host_nvnMultisampleStateSetSampleLocationsEnable", hs_MultisampleStateSampleLocationsEnable);
NX_HLE("host_nvnMultisampleStateGetSampleLocationsEnable", hg_MultisampleStateSampleLocationsEnable);
NX_HLE("host_nvnMultisampleStateSetSampleLocationsGridEnable", hs_MultisampleStateSampleLocationsGridEnable);
NX_HLE("host_nvnMultisampleStateGetSampleLocationsGridEnable", hg_MultisampleStateSampleLocationsGridEnable);
NX_HLE("host_nvnMultisampleStateSetSampleLocations", hs_MultisampleStateSampleLocations);
NX_HLE("host_nvnMultisampleStateGetSampleLocationsGrid", hg_MultisampleStateSampleLocationsGrid);
NX_HLE("host_nvnPolygonStateSetDefaults", hd_PolygonState);
NX_HLE("host_nvnPolygonStateSetCullFace", hs_PolygonStateCullFace);
NX_HLE("host_nvnPolygonStateGetCullFace", hg_PolygonStateCullFace);
NX_HLE("host_nvnPolygonStateSetFrontFace", hs_PolygonStateFrontFace);
NX_HLE("host_nvnPolygonStateGetFrontFace", hg_PolygonStateFrontFace);
NX_HLE("host_nvnPolygonStateSetPolygonMode", hs_PolygonStatePolygonMode);
NX_HLE("host_nvnPolygonStateGetPolygonMode", hg_PolygonStatePolygonMode);
NX_HLE("host_nvnPolygonStateSetPolygonOffsetEnables", hs_PolygonStatePolygonOffsetEnables);
NX_HLE("host_nvnPolygonStateGetPolygonOffsetEnables", hg_PolygonStatePolygonOffsetEnables);
NX_HLE("host_nvnVertexAttribStateSetDefaults", hd_VertexAttribState);
NX_HLE("host_nvnVertexAttribStateSetFormat", hs_VertexAttribStateFormat);
NX_HLE("host_nvnVertexAttribStateGetFormat", hg_VertexAttribStateFormat);
NX_HLE("host_nvnVertexAttribStateSetStreamIndex", hs_VertexAttribStateStreamIndex);
NX_HLE("host_nvnVertexAttribStateGetStreamIndex", hg_VertexAttribStateStreamIndex);
NX_HLE("host_nvnVertexStreamStateSetDefaults", hd_VertexStreamState);
NX_HLE("host_nvnVertexStreamStateSetStride", hs_VertexStreamStateStride);
NX_HLE("host_nvnVertexStreamStateGetStride", hg_VertexStreamStateStride);
NX_HLE("host_nvnVertexStreamStateSetDivisor", hs_VertexStreamStateDivisor);
NX_HLE("host_nvnVertexStreamStateGetDivisor", hg_VertexStreamStateDivisor);
NX_HLE("host_nvnTextureViewSetDefaults", hd_TextureView);
NX_HLE("host_nvnTextureViewSetLevels", hs_TextureViewLevels);
NX_HLE("host_nvnTextureViewGetLevels", hg_TextureViewLevels);
NX_HLE("host_nvnTextureViewSetLayers", hs_TextureViewLayers);
NX_HLE("host_nvnTextureViewGetLayers", hg_TextureViewLayers);
NX_HLE("host_nvnTextureViewSetFormat", hs_TextureViewFormat);
NX_HLE("host_nvnTextureViewGetFormat", hg_TextureViewFormat);
NX_HLE("host_nvnTextureViewSetSwizzle", hs_TextureViewSwizzle);
NX_HLE("host_nvnTextureViewGetSwizzle", hg_TextureViewSwizzle);
NX_HLE("host_nvnTextureViewSetDepthStencilMode", hs_TextureViewDepthStencilMode);
NX_HLE("host_nvnTextureViewGetDepthStencilMode", hg_TextureViewDepthStencilMode);
NX_HLE("host_nvnTextureViewSetTarget", hs_TextureViewTarget);
NX_HLE("host_nvnTextureViewGetTarget", hg_TextureViewTarget);
NX_HLE("host_nvnTextureViewCompare", hc_TextureView);
NX_HLE("host_nvnWindowBuilderGetNumTextures", hb_WindowBuilderGetNumTextures);
NX_HLE("host_nvnWindowBuilderGetTexture", hb_WindowBuilderGetTexture);
NX_HLE("host_nvnWindowBuilderGetTextures", hb_WindowBuilderGetTextures);
NX_HLE("host_nvnTextureBuilderGetPackagedTextureLayout", hb_TextureBuilderGetPackagedTextureLayout);

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
    if (!fn) fatal("nvn: guest_call of a null function (a proc the driver does not have)");
    uint64_t saved[31];
    std::memcpy(saved, c.x, sizeof saved);
    size_t k = 0;
    for (uint64_t a : args) c.x[k++] = a;
    if (HostFn host = host_fn_for(fn))
        host(c);  // a host builder proc: run directly, no trip through the guest
    else
        call(c, fn);
    uint64_t r = c.x[0];
    std::memcpy(c.x, saved, sizeof saved);
    return r;
}

float guest_call_f(CPU& c, uint64_t fn, std::initializer_list<uint64_t> args) {
    if (!fn) fatal("nvn: guest_call_f of a null function (a proc the driver does not have)");
    uint64_t saved[31];
    std::memcpy(saved, c.x, sizeof saved);
    V128 v0 = c.v[0];
    size_t k = 0;
    for (uint64_t a : args) c.x[k++] = a;
    if (HostFn host = host_fn_for(fn))
        host(c);
    else
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

uint64_t driver_proc(CPU& c, const char* name) { return real_proc_slow(c, name); }

// A proc for the runtime's own callers (the state-object getters, one per bound state): the SDK driver's, else the
// host's version (host builders), else a function that stops with the proc's name. Never 0.
uint64_t real_proc(CPU& c, const char* name) {
    // The hot callers pass string literals: a per-thread cache by the literal's address, verified by content, instead
    // of a global mutex and a string-keyed tree at every call (the game's recording threads contended on that lock).
    struct Hit {
        std::string name;
        uint64_t proc = 0;
    };
    thread_local std::unordered_map<const char*, Hit> fast;
    if (auto it = fast.find(name); it != fast.end() && it->second.name == name) return it->second.proc;
    uint64_t found = real_proc_slow(c, name);
    if (!found) found = host_proc_address(name);
    if (!found) {
        static std::mutex noted_lock;
        static std::set<std::string> noted;
        bool first;
        {
            std::lock_guard lk(noted_lock);
            first = noted.insert(name).second;
        }
        if (first) log("nvn: real_proc(%s): no driver and no host version: calling it stops with this name", name);
        const std::string stub = std::string("missing_") + name;
        found = hle_address_or(stub.c_str(), missing_driver_proc);
    }
    fast[name] = {name, found};
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
