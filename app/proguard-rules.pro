# JNI: native methods are resolved by name, and native code calls back into
# NativeEnhancer.dispatchProgress via GetStaticMethodID.
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep class com.imagecleaner.app.core.NativeEnhancer { *; }

# OkHttp / Okio
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn org.conscrypt.**
