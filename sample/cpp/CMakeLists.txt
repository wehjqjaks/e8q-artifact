add_executable(e8q_index e8q_index.cpp)
add_executable(me8q_index me8q_index.cpp)

foreach(E8Q_SAMPLE_TARGET e8q_index me8q_index)
    target_link_libraries(${E8Q_SAMPLE_TARGET} PRIVATE e8q_core)
    e8q_enable_warnings(${E8Q_SAMPLE_TARGET})
endforeach()
