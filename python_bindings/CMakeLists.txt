set(PYBIND11_FINDPYTHON ON)
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

if(CMAKE_VERSION VERSION_LESS 3.18)
    find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_e8qlib
    module.cpp
    e8q_bindings.cpp
    me8q_bindings.cpp
)

target_include_directories(_e8qlib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
e8q_enable_warnings(_e8qlib)

target_link_libraries(_e8qlib PRIVATE
    e8q_core
    Python::NumPy
)

install(TARGETS _e8qlib DESTINATION e8qlib)
install(FILES __init__.py DESTINATION e8qlib)
