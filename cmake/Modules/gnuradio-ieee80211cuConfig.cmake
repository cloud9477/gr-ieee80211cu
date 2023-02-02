find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_IEEE80211CU gnuradio-ieee80211cu)

FIND_PATH(
    GR_IEEE80211CU_INCLUDE_DIRS
    NAMES gnuradio/ieee80211cu/api.h
    HINTS $ENV{IEEE80211CU_DIR}/include
        ${PC_IEEE80211CU_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_IEEE80211CU_LIBRARIES
    NAMES gnuradio-ieee80211cu
    HINTS $ENV{IEEE80211CU_DIR}/lib
        ${PC_IEEE80211CU_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-ieee80211cuTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_IEEE80211CU DEFAULT_MSG GR_IEEE80211CU_LIBRARIES GR_IEEE80211CU_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_IEEE80211CU_LIBRARIES GR_IEEE80211CU_INCLUDE_DIRS)
