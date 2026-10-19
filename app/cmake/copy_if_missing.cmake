# cmake -DSRC=<file> -DDST=<file> -P copy_if_missing.cmake: copies SRC to DST only when DST does not exist.
if(NOT EXISTS "${DST}")
    configure_file("${SRC}" "${DST}" COPYONLY)
endif()
