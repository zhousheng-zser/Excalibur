APP_OPTIM := release  
APP_ABI := arm64-v8a
APP_CPPFLAGS += -fPIC -std=c++17 -fpermissive -frtti -fexceptions
#APP_GNUSTL_FORCE_CPP_FEATURES := exceptions rtti  
APP_STL := c++_static
APP_PLATFORM := android-19