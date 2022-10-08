#pragma once

#ifdef AIPILOT_EXPORTS
#define CREATE_RELEASE_API extern "C" __declspec(dllexport)
#else
#define CREATE_RELEASE_API extern "C" __declspec(dllimport)
#endif

#include "AIPilot.h"
#include "AIPilot_TeamIntelligame.h"

CREATE_RELEASE_API AIPilot* CreateInstance(unsigned short index);

CREATE_RELEASE_API void ReleaseInstance(unsigned short index);