#include <emscripten/bind.h>
#include <emscripten/val.h>
using namespace emscripten;

val test_version() {
    return val("TEST-VERSION-2025-12-22");
}

EMSCRIPTEN_BINDINGS(test_module) {
    emscripten::function("test_version", &test_version);
}
