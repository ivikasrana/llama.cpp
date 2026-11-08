#pragma once


#define INNERQ_MAX_CHANNELS 128

#ifdef GGML_BACKEND_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#      define TURBO_IQ_API __declspec(dllexport)
#    else
#      define TURBO_IQ_API __declspec(dllimport)
#    endif
#  else
#    define TURBO_IQ_API __attribute__((visibility("default")))
#  endif
#else
#  define TURBO_IQ_API
#endif

TURBO_IQ_API extern bool  g_innerq_finalized;
TURBO_IQ_API extern float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

void turbo_innerq_publish(const float * scale_inv, int group_size);

const float * ggml_cuda_turbo_innerq_scale_inv(void);

TURBO_IQ_API bool turbo_innerq_needs_tensor_update(void);

TURBO_IQ_API void turbo_innerq_mark_tensor_updated(void);
