#include "plugin.h"
#include "PlatformBase.h"
#include "PluginAPI.h"

#include <string>
#include <sstream>

using namespace std;

static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static PluginAPI* api = nullptr;

extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetLastMessage() {
	if (api == nullptr) { return "NO API"; }
	return api->GetLastMessage();
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsAPIReady() {
	return api != nullptr;
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API LoadModel(const char* filepath) {
	if (api == nullptr) { return false; }
	return api->LoadModel(filepath);

}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetNbPov(int nb_pov) {
	if (api == nullptr) { return; }
	api->SetNbPov(nb_pov);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetPovParameters(int pov, int width, int height) {
	if (api == nullptr) { return; }
	api->SetPovParameters(pov, width, height);
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsInitialized() {
	if (api == nullptr) { return false; }
	return api->IsInitialized();
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API DrawSync() {
	if (api == nullptr) { return false; }
	return api->Draw();
}

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsDrawn() {
	if (api == nullptr) { return false; }
	return api->IsDrawn();
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetDrawParameters(int pov, float* position, float* rotation, float* proj, float fovy, float* frustums) {
	if (api == nullptr) { return; }
	api->SetDrawParameters(pov, position, rotation, proj, fovy, frustums);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetLightingParameters(int H, int W, float* position, float* rotation, float* proj, float fovy, float* frustums) {
	if (api == nullptr) { return; }
	api->SetLightingParameters(H, W, position, rotation, proj, fovy, frustums);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetSceneSize(float* scene_min, float* scene_max) {
	if (api == nullptr) { return; }
	api->splat.GetSceneSize(scene_min, scene_max);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetSimParams(float frame_dt, 
	float dt, 
	float gravity, 
	float damping_coeffient,
	int rest_iter, 
	float collision_stiffness, 
	float minimal_dist, 
	float collision_detection_iter_interval,
	float shadow_eps,
	float shadow_factor,
	int zup, 
	float ground_height, 
	float global_scale, 
	float *global_offset, 
	float *global_rotation, 
	int num_objects, 
	float *object_offsets,
	float *object_materials,
	int disable_LG_interp,
	int quasi_static,
	int max_quasi_sequence,
	int enable_export_frames,
    int max_export_sequence,
	int boundary) {
	if (api == nullptr) { return; }
	api->splat.SetSimParams(frame_dt, 
		dt, 
		gravity, 
		damping_coeffient,
		rest_iter, 
		collision_stiffness,
		minimal_dist, 
		collision_detection_iter_interval,
		shadow_eps, 
		shadow_factor,
		zup, 
		ground_height, 
		global_scale, 
		global_offset, 
		global_rotation, 
		num_objects, 
		object_offsets,
		object_materials,
		disable_LG_interp,
		quasi_static,
		max_quasi_sequence,
		enable_export_frames,
    	max_export_sequence,
		boundary);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetController(
	float controller_radius,
	float *lpos, float *lrot, int ltriggered,
	float *rpos, float *rrot, int rtriggered) {
	if (api == nullptr) { return; }
	api->splat.SetController(
		controller_radius,
		lpos, lrot, ltriggered,
		rpos, rrot, rtriggered);
}

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetNbSplat() {
	if (api == nullptr) { return 0; }
	return api->splat.total_gs;
}

extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetTextureNativePointer(int pov) {
	if (api == nullptr) { return nullptr; }
	api->GetTextureNativePointer(pov);
}

//Render event callback, called by GL.IssuePluginEvent on render thread.
static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	if (api != nullptr) {
		api->OnRenderEvent(eventID);
	}
}


//Device event callback
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	// Create graphics API implementation upon initialization
	if (eventType == UnityGfxDeviceEventType::kUnityGfxDeviceEventInitialize)
	{
		if (api != nullptr) { delete api; }
		api = PluginAPI::Create(s_Graphics->GetRenderer(), s_UnityInterfaces);
	}

	if (api != nullptr) {
		api->OnGraphicsDeviceEvent(eventType);
	}

	// Cleanup graphics API implementation upon shutdown
	if (eventType == UnityGfxDeviceEventType::kUnityGfxDeviceEventShutdown)
	{
		if (api != nullptr) { delete api; api = nullptr; }
	}
}

// Section to interface unity.
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
	s_UnityInterfaces = unityInterfaces;
	s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
	s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

#if SUPPORT_VULKAN
	if (s_Graphics->GetRenderer() == kUnityGfxRendererNull)
	{
		extern void RenderAPI_Vulkan_OnPluginLoad(IUnityInterfaces*);
		RenderAPI_Vulkan_OnPluginLoad(unityInterfaces);
	}
#endif // SUPPORT_VULKAN

	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	OnGraphicsDeviceEvent(UnityGfxDeviceEventType::kUnityGfxDeviceEventInitialize);
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload()
{
	s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API Save() {
	if (api == nullptr) { return; }
	api->splat.Save();
}