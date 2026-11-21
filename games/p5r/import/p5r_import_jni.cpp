// The import on the phone (libp5rimport.so, loaded by ImportActivity in its own process ":import"): the player's NSP or
// XCI and keys -> the game file opened and checked (title Farming Simulator Nintendo Switch Edition, 0100D04004176000).
// This version does NOT extract or translate shaders (that was Persona 5 Royal's GFD shader caches in .CPK/.GSC):
// it only creates <data>/spirv and writes import.txt. The game itself is read from the NSP at every start
// (runtime game_image.cpp): nothing else is copied.
//   Importer.run(nspFd, nspName, keysPath, titleKeysPath, dataDir, workDir, jobs) -> 0, or an error (Importer.error())
//   Importer.progress() -> {phase, done, total}: 1 opening, 4 finishing, 5 done
#include <dirent.h>
#include <fcntl.h>
#include <jni.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "nxgame.h"

namespace {

// false: only Farming Simulator is accepted. true: any valid NSP/XCI is imported whatever its Title ID.
constexpr bool kAllowAnyTitle = false;

static std::string tid_str(uint64_t t) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)t);
    return buf;
}

// The base title (0100D04004176000) and its update (0100D04004176800) share the same base once the low 12 bits are
// cleared, so a patched dump (1.5.0.1) is accepted too.
bool is_supported_title(uint64_t t) {
    if (kAllowAnyTitle) return true;
    uint64_t base = t & ~0xFFFull;
    return base == 0x0100D04004176000ull;  // Farming Simulator Nintendo Switch Edition
}

std::atomic<int> g_phase{0}, g_done{0}, g_total{0};
std::mutex g_error_lock;
std::string g_error;

void set_error(const std::string& e) {
    std::lock_guard lk(g_error_lock);
    g_error = e;
}

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

void remove_tree(const std::string& path) {
    if (DIR* d = opendir(path.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            std::string p = path + "/" + n;
            struct stat st{};
            if (lstat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) remove_tree(p);
            else unlink(p.c_str());
        }
        closedir(d);
    }
    rmdir(path.c_str());
}

int fail(const std::string& e) {
    set_error(e);
    g_phase = 0;
    return 1;
}

int run(int nsp_fd, const std::string& nsp_name, const std::string& keys, const std::string& title_keys,
        const std::string& data_dir, const std::string& work_dir, int jobs) {
    (void)work_dir;
    (void)jobs;
    g_phase = 1, g_done = 0, g_total = 0;
    set_error("");
    // 1. the game file and the keys
    std::vector<nxi::InputFile> in;
    in.push_back({nsp_name, nsp_fd});
    int kfd = open(keys.c_str(), O_RDONLY | O_CLOEXEC);
    if (kfd < 0) return fail("cannot read prod.keys");
    in.push_back({"prod.keys", kfd});
    int tfd = title_keys.empty() ? -1 : open(title_keys.c_str(), O_RDONLY | O_CLOEXEC);
    if (tfd >= 0) in.push_back({"title.keys", tfd});
    std::string err;
    auto game = nxi::GameImage::open(in, 0, err);
    close(kfd);
    if (tfd >= 0) close(tfd);
    if (!game) return fail(err);
    if (!is_supported_title(game->title_id()))
        return fail("unsupported Title ID: " + tid_str(game->title_id()) + " (expected Farming Simulator Switch Edition, 0100D04004176000)");
    auto main_it = game->exefs().find("main");
    if (main_it == game->exefs().end() || main_it->second.size() < 0x48) return fail("the game file has no main");
    char id[17];
    for (int i = 0; i < 8; i++) std::snprintf(id + 2 * i, 3, "%02X", main_it->second[0x40 + i]);
    // The build id is only recorded: no build is blocked.

    // 2. no shader extraction or translation: an empty <data>/spirv and a note of what was imported
    g_phase = 4;
    const std::string out = data_dir + "/spirv";
    remove_tree(out);  // stale files of another game (Persona 5 Royal's translated shaders) must not stay
    if (mkdir(out.c_str(), 0700) != 0 && errno != EEXIST) return fail("cannot create " + out);
    FILE* f = std::fopen((out + "/import.txt").c_str(), "w");
    if (!f) return fail("cannot write " + out + "/import.txt");
    std::fprintf(f, "%s\n%s\n", nsp_name.c_str(), id);
    std::fclose(f);
    g_phase = 5;
    return 0;
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL Java_io_github_p5rrecomp_Importer_run(JNIEnv* env, jclass, jint nsp_fd, jstring nsp_name,
                                                             jstring keys, jstring title_keys, jstring data_dir,
                                                             jstring work_dir, jint jobs) {
    return run(nsp_fd, jstr(env, nsp_name), jstr(env, keys), jstr(env, title_keys), jstr(env, data_dir),
               jstr(env, work_dir), jobs);
}

JNIEXPORT jintArray JNICALL Java_io_github_p5rrecomp_Importer_progress(JNIEnv* env, jclass) {
    int v[3] = {g_phase.load(), g_done.load(), g_total.load()};
    jintArray a = env->NewIntArray(3);
    env->SetIntArrayRegion(a, 0, 3, v);
    return a;
}

JNIEXPORT jstring JNICALL Java_io_github_p5rrecomp_Importer_error(JNIEnv* env, jclass) {
    std::lock_guard lk(g_error_lock);
    return env->NewStringUTF(g_error.c_str());
}

}  // extern "C"
