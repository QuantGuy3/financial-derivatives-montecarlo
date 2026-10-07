# Interfaz web local: servidor C++ (cpp-httplib + nlohmann/json) y recursos embebidos en el binario.

set(MC_WEB_SPECS
    "${PROJECT_SOURCE_DIR}/gui/web/index.html|/index.html"
    "${PROJECT_SOURCE_DIR}/gui/web/favicon.svg|/favicon.svg"
    "${PROJECT_SOURCE_DIR}/gui/web/assets/style.css|/assets/style.css"
    "${PROJECT_SOURCE_DIR}/gui/web/assets/i18n.js|/assets/i18n.js"
    "${PROJECT_SOURCE_DIR}/gui/web/assets/api.js|/assets/api.js"
    "${PROJECT_SOURCE_DIR}/gui/web/assets/charts.js|/assets/charts.js"
    "${PROJECT_SOURCE_DIR}/gui/web/assets/app.js|/assets/app.js"
    "${PROJECT_SOURCE_DIR}/third_party/echarts/echarts.min.js|/vendor/echarts.min.js"
)
set(MC_WEB_FILES "")
foreach(spec IN LISTS MC_WEB_SPECS)
    string(REPLACE "|" ";" parts "${spec}")
    list(GET parts 0 f)
    list(APPEND MC_WEB_FILES "${f}")
endforeach()

set(MC_WEB_ASSETS_CPP "${CMAKE_CURRENT_BINARY_DIR}/web_assets.cpp")
string(REPLACE ";" "\;" MC_WEB_SPECS_ESC "${MC_WEB_SPECS}")
add_custom_command(
    OUTPUT "${MC_WEB_ASSETS_CPP}"
    COMMAND ${CMAKE_COMMAND} "-DOUT=${MC_WEB_ASSETS_CPP}" "-DSPECS=${MC_WEB_SPECS_ESC}"
            -P "${PROJECT_SOURCE_DIR}/cmake/embed_assets.cmake"
    DEPENDS ${MC_WEB_FILES} "${PROJECT_SOURCE_DIR}/cmake/embed_assets.cmake"
    COMMENT "Embebiendo la interfaz web"
    VERBATIM)

add_library(mc_gui_lib STATIC
    json_codec.cpp
    jobs.cpp
    server.cpp
    "${MC_WEB_ASSETS_CPP}")
target_include_directories(mc_gui_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# cpp-httplib (<httplib.h>) y nlohmann/json (<nlohmann/json.hpp>) vendorizados; SYSTEM silencia sus avisos
target_include_directories(mc_gui_lib SYSTEM PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib
    ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(mc_gui_lib PUBLIC mc_engine PRIVATE mc_warnings)
if(WIN32)
    target_link_libraries(mc_gui_lib PUBLIC ws2_32)
    target_compile_definitions(mc_gui_lib PUBLIC _WIN32_WINNT=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_executable(mc_gui main.cpp)
target_link_libraries(mc_gui PRIVATE mc_gui_lib)
if(WIN32 AND MINGW)
    # Ventana de aplicación: sin consola extra no hace falta; se deja la consola para ver la URL/errores
endif()
