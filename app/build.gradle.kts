import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.imagecleaner.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.imagecleaner.app"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            // arm64 covers virtually all active Android devices. Add armeabi-v7a / x86_64
            // by running tools/setup_ncnn.sh (it fetches ABIs for every architecture).
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fvisibility=hidden", "-fopenmp")
                arguments += listOf("-DANDROID_STL=c++_static")
            }
        }

        // Override at publish time: -PmodelsBaseUrl=https://...
        buildConfigField(
            "String",
            "MODELS_BASE_URL_OVERRIDE",
            "\"${(project.findProperty("modelsBaseUrl") as String?) ?: ""}\""
        )
    }

    // Optional release signing via keystore.properties (see README, never commit the keystore)
    val keystorePropsFile = rootProject.file("keystore.properties")
    val hasReleaseKeystore = keystorePropsFile.exists()
    signingConfigs {
        if (hasReleaseKeystore) {
            create("release") {
                val props = Properties().apply { load(keystorePropsFile.inputStream()) }
                storeFile = file(props.getProperty("storeFile"))
                storePassword = props.getProperty("storePassword")
                keyAlias = props.getProperty("keyAlias")
                keyPassword = props.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            if (hasReleaseKeystore) signingConfig = signingConfigs.getByName("release")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    // ncnn 20240820 prebuilts link against a newer libc++ (verbose_abort etc.);
    // NDK r25 fails at link time, r27+ works.
    ndkVersion = "29.0.14206865"

    externalNativeBuild {
        cmake {
            path = file("../native/CMakeLists.txt")
            version = "3.10.2.4988404"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    lint {
        // Some local SDK installs contain preview platforms like android-37.2
        // whose version string crashes lint's parser. CI runs :app:lint explicitly.
        checkReleaseBuilds = false
        abortOnError = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.exifinterface)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.okhttp)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
}
