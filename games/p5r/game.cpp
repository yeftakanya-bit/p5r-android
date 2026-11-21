// Farming Simulator Nintendo Switch Edition (0100D04004176000, 1.5.0.1, GIANTS Engine): what the shared runtime
// needs to know about the game. Every Persona 5 Royal specific part (fixed main+offset hooks, the frame interval
// setter, the culling job, the HUD alpha hacks) was removed: a GIANTS Engine NSO has other code at those offsets.
// The game draws through the standard NVN path implemented by the runtime.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

#include <nx/game.h>
#include <nx/runtime.h>

#include "platform.h"

namespace nx {

// main_build_id is nullptr: main.cpp skips the build id check altogether.
// id, package and data directory keep the values of the original fork (the Android package and the data folder are
// referenced outside this file). system_resource_size 0 is P5R's main.npdm value; if the game aborts early in
// nn::os / nn::init, check the game's own main.npdm.
const game::Info game::kInfo = {"p5r", "Farming Simulator Nintendo Switch Edition", 0x0100D04004176000ull,
                                "io.github.p5rrecomp", "private/p5r", 0 /* system resource size */,
                                nullptr /* main_build_id: no build id check */};

namespace {

// glslc, NVN's run-time GLSL compiler. The runtime has no GLSL compiler: Initialize fails and nothing is compiled, so
// a game that ships its shaders precompiled (as NVN games normally do) never notices. If the log shows this being
// called, the game compiles GLSL at run time and needs a real implementation in the runtime.
void glslc_fail(CPU& c) {
    log("game: %s called: run-time shader compilation is not supported", hle_current_name());
    c.x[0] = 0;
}

void glslc_nop(CPU& c) { c.x[0] = 0; }

// Copy every fputs line the game writes into the log (its assertion handlers write to stderr, which goes nowhere on
// Android), then do the real one.
void fputs_logged(CPU& c) {
    static const uint64_t real = find_symbol("fputs");
    const char* s = reinterpret_cast<const char*>(c.x[0]);
    if (s) {
        std::string line(s, strnlen(s, 1024));
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        log("[guest fputs] %s", line.c_str());
    }
    if (real) call(c, real);
    else c.x[0] = 0;
}

}  // namespace

NX_HLE("fputs", fputs_logged);

NX_HLE("glslcInitialize", glslc_fail);
NX_HLE("glslcCompile", glslc_fail);
NX_HLE("glslcCompilePreSpecialized", glslc_fail);
NX_HLE("glslcCompileSpecialized", glslc_fail);
NX_HLE("glslcFinalize", glslc_nop);
NX_HLE("glslcSetAllocator", glslc_nop);
NX_HLE("glslcGetDefaultOptions", glslc_nop);
NX_HLE("glslcGetVersion", glslc_nop);

// No game specific hooks: nothing is patched in the game's code and no P5R environment switches are set
// (NX_OPAQUE_FS, NX_HUD_KEEP_ALPHA, NX_HUD_1080, NX_IMAGE_STORAGE and NX_GAME_LANGUAGES are left to the runtime's
// own defaults, or to config.txt).
void game::after_init() {
    game::renderer_start();
}

static constexpr const char* kPerfBuildTag = "universal-nvn-opt";

void game::renderer_start() {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    log("perf build: %s", kPerfBuildTag);
#if defined(__ANDROID__)
    // Universal Android defaults (setenv with overwrite = 0: config.txt or the launcher can still change them):
    // - Per-image deferred clears: clears fold into VK_ATTACHMENT_LOAD_OP_CLEAR instead of many standalone
    //   vkCmdClear*Image transfer clears.
    setenv("NX_CLEARS_V2", "1", 0);
    // - Skip redundant CPU write-backs of GPU compute written buffers (experimental: if meshes or skinned objects
    //   look wrong, set NX_EXP_NO_WB=0 in config.txt).
    setenv("NX_EXP_NO_WB", "1", 0);
    // - Do not force-clear color attachments on every new render pass without an explicit NVN clear command.
    setenv("NX_LOAD_CLEAR", "0", 0);
    // - Keep the default NX_SUBMIT_CHUNK_DRAWS and NX_MALI_FP16 off: the runtime's per-GPU workarounds
    //   (Adreno 8xx, Mali) are decided in the runtime, not here.
#endif
}

void game::first_frame() {}

}  // namespace nx
