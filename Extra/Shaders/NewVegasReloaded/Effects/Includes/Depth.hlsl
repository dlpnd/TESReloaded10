// A collection of functions that allow to query depth of a given pixel and also to reconstruct/project a point from screen to view space
// requires the shader to get access to the TESR_DepthBuffer sampler before the include.

float4x4 TESR_RealProjectionTransform;
float4x4 TESR_RealInvProjectionTransform;
float4 TESR_DepthConstants;
float4 TESR_CameraData;
static const float nearZ = TESR_CameraData.x;
static const float farZ = TESR_CameraData.y;
static const float Q = farZ/(farZ - nearZ);

float readDepth(in float2 coord : TEXCOORD0)
{
	float Depth = tex2D(TESR_DepthBuffer, coord).x;;
    float ViewZ = (-nearZ *Q) / (Depth - Q);
	return ViewZ;
}

float3 reconstructPosition(float2 uv)
{
	float4 screenpos = float4(uv * 2.0 - 1.0f, tex2D(TESR_DepthBuffer, uv).x, 1.0f);
	screenpos.y = -screenpos.y;
	float4 viewpos = mul(screenpos, TESR_RealInvProjectionTransform);
	viewpos.xyz /= viewpos.w;
	return viewpos.xyz;
}

float3 projectPosition(float3 position){
	float4 projection = mul(float4 (position, 1.0), TESR_RealProjectionTransform);
	projection.xyz /= projection.w;
	projection.x = projection.x * 0.5 + 0.5;
	projection.y = 0.5 - 0.5 * projection.y;

	return projection.xyz;
}
