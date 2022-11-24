// Ambient Occlusion fullscreen shader for Oblivion/Skyrim Reloaded

#define viewao 0

float4x4 TESR_ProjectionTransform;
float4 TESR_ReciprocalResolution;
float4 TESR_AmbientOcclusionAOData;
float4 TESR_AmbientOcclusionData;
float4 TESR_DepthConstants;
float4 TESR_CameraData;
float4 TESR_FogData;

sampler2D TESR_RenderedBuffer : register(s0) = sampler_state { ADDRESSU = CLAMP; ADDRESSV = CLAMP; MAGFILTER = LINEAR; MINFILTER = LINEAR; MIPFILTER = LINEAR; };
sampler2D TESR_DepthBuffer : register(s1) = sampler_state { ADDRESSU = CLAMP; ADDRESSV = CLAMP; MAGFILTER = LINEAR; MINFILTER = LINEAR; MIPFILTER = LINEAR; };
sampler2D TESR_SourceBuffer : register(s2) = sampler_state { ADDRESSU = CLAMP; ADDRESSV = CLAMP; MAGFILTER = LINEAR; MINFILTER = LINEAR; MIPFILTER = LINEAR; };

static const int samples = 5;
static const int rings = 2;
static const float nearZ = TESR_CameraData.x;
static const float farZ = TESR_CameraData.y;
static const float aspectRatio = TESR_CameraData.z;
static const float fov = TESR_CameraData.w;
static const float Zmul = TESR_DepthConstants.z;
static const float Zdiff = TESR_DepthConstants.w;
static const float depthRange = TESR_DepthConstants.w;
static const float depthConstant = depthRange / 20000;
static const float aoMultiplier = 500.0;
static const float angleBiasRadians = radians(TESR_AmbientOcclusionData.x);
static const float cosAngleBias = cos(angleBiasRadians);
static const float sinAngleBias = sin(angleBiasRadians);
static const float2 OffsetMaskH = float2(1.0f, 0.0f);
static const float2 OffsetMaskV = float2(0.0f, 1.0f);
static const float PI = 3.14159265;
static const int cKernelSize = 12;

static const float BlurWeights[cKernelSize] = 
{
	0.057424882f,
	0.058107773f,
	0.061460144f,
	0.071020611f,
	0.088092873f,
	0.106530916f,
	0.106530916f,
	0.088092873f,
	0.071020611f,
	0.061460144f,
	0.058107773f,
	0.057424882f
};
 
static const float2 BlurOffsets[cKernelSize] = 
{
	float2(-6.0f * TESR_ReciprocalResolution.x, -6.0f * TESR_ReciprocalResolution.y),
	float2(-5.0f * TESR_ReciprocalResolution.x, -5.0f * TESR_ReciprocalResolution.y),
	float2(-4.0f * TESR_ReciprocalResolution.x, -4.0f * TESR_ReciprocalResolution.y),
	float2(-3.0f * TESR_ReciprocalResolution.x, -3.0f * TESR_ReciprocalResolution.y),
	float2(-2.0f * TESR_ReciprocalResolution.x, -2.0f * TESR_ReciprocalResolution.y),
	float2(-1.0f * TESR_ReciprocalResolution.x, -1.0f * TESR_ReciprocalResolution.y),
	float2( 1.0f * TESR_ReciprocalResolution.x,  1.0f * TESR_ReciprocalResolution.y),
	float2( 2.0f * TESR_ReciprocalResolution.x,  2.0f * TESR_ReciprocalResolution.y),
	float2( 3.0f * TESR_ReciprocalResolution.x,  3.0f * TESR_ReciprocalResolution.y),
	float2( 4.0f * TESR_ReciprocalResolution.x,  4.0f * TESR_ReciprocalResolution.y),
	float2( 5.0f * TESR_ReciprocalResolution.x,  5.0f * TESR_ReciprocalResolution.y),
	float2( 6.0f * TESR_ReciprocalResolution.x,  6.0f * TESR_ReciprocalResolution.y)
};
 
struct VSOUT
{
	float4 vertPos : POSITION;
	float2 UVCoord : TEXCOORD0;
};
 
struct VSIN
{
	float4 vertPos : POSITION0;
	float2 UVCoord : TEXCOORD0;
};
 
VSOUT FrameVS(VSIN IN)
{
	VSOUT OUT = (VSOUT)0.0f;
	OUT.vertPos = IN.vertPos;
	OUT.UVCoord = IN.UVCoord;
	return OUT;
}
 
// from https://gist.github.com/keijiro/ee7bc388272548396870
float nrand(float2 uv)
{
    return frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

float2 rand(in float2 uv)
{
	float noiseX = frac(sin(dot(uv ,float2(12.9898, 78.233))) * 43758.5453);
	float noiseY = frac(sin(dot(uv ,float2(12.9898, 78.233) * 2.0)) * 43758.5453);
	return float2(noiseX, noiseY) * 0.004;
}
 
float readDepth(in float2 coord : TEXCOORD0)
{
	float posZ = tex2D(TESR_DepthBuffer, coord).x;
	float a = farZ / Zdiff;
	float b = -Zmul/Zdiff;
	posZ = b/(posZ - a);

	return posZ;
}
 
float3 getPosition(in float2 uv)
{
    float eye_z = readDepth(uv);

    uv = (uv * float2(2.0, -2.0) - float2(1.0, -1.0));
    float3 pos = float3(uv * g_InvFocalLen * eye_z, eye_z );
    return pos;
}

float2 packDepth(float depth)
{
    return float2(depth, frac(depth * 255.0 - 0.5));
}
 
float unpackDepth(float2 depth)
{
    return depth.x + ((depth.y - 0.5) / 255.0);
}
 
float compareDepths(float depth1, float depth2)
{
	float diff = sqrt(saturate(1.0 - (depth1 - depth2) / (TESR_AmbientOcclusionAOData.w / depthRange)));
	float dist = (depth1 - depth2) * depthConstant;
	float ao = min(1, dist * aoMultiplier) * diff;
	float depth1Pos = depth1 * depthRange;
	
	if (depth1Pos >= 8000) ao *= saturate(lerp(1, 0, (depth1Pos - 8000) / 12500));
 
	return float(ao);
}

float3 reconstructPositionFromFOV(float2 uv){
	float3 position = float3(uv*2.0f-1.0f, readDepth(uv));
	position.xy *= position.z * (radians(fov*0.5));
	position.x *= aspectRatio;

	return position;
}


float3 projectPositionFromFOV(float3 position){
	float2 screenCoords = position.xy;
	screenCoords /= radians(fov*0.5);
	screenCoords.x /= aspectRatio;
	screenCoords /= position.z;
	
	return float3(screenCoords * 0.5 + 0.5, 1.0);
}

float3 reconstructPositionFromMatrix(float2 uv){	
	float x = uv.x * 2.0f - 1.0f;
	float y = uv.y * -2.0f + 1.0f;
	float z = readDepth(uv);

	float4 screenpos = float4(x, y, z, 1.0f);
	float4 viewpos = mul(screenpos, TESR_RealInvProjectionTransform);
	viewpos.xyz /= viewpos.w;
	return viewpos;
}


float3 reconstructPosition(float2 uv)
{
	return reconstructPositionFromFOV(uv);

	return reconstructPositionFromMatrix(uv);
}


float3 projectPosition(float3 position){

	return projectPositionFromFOV(position);

	float4 projection = mul(float4 (position, 1.0), TESR_RealProjectionTransform);
	projection.xyz /= projection.w;
	projection.x = projection.x * 0.5 + 0.5;
	projection.y = 0.5 + 0.5 * projection.y;

	return projection;
}

float3 GetNormal( float2 coord)
{
	float2 uv = coord; // center
	float2 uv1 = uv + float2(1, 0) * TESR_ReciprocalResolution.x; // right 
	float2 uv2 = uv + float2(0, 1) * TESR_ReciprocalResolution.y; // top

	float3 P0 = reconstructPosition(uv);
	float3 P1 = reconstructPosition(uv1);
	float3 P2 = reconstructPosition(uv2);

	return normalize(cross(P2 - P0, P1 - P0));
}


float4 SSAO(VSOUT IN) : COLOR0
{
	float2 coord = IN.UVCoord;
	float3 origin = reconstructPosition(coord);
	float3 screenSpaceSample = projectPosition (origin);
	float sampleDepth = readDepth(screenSpaceSample.xy);
	float depth = readDepth(coord);
	float3 normal = GetNormal(coord);

	uint kernelSize = 12;
	float3 kernel[12];
	float uRadius = 300;

	for (uint i = 0; i < kernelSize; ++i) {
		// generate random samples in a unit sphere
		kernel[i] = float3 (nrand(float2(-1.0f, 1.0f)), nrand(float2(-1.0f, 1.0f)), nrand(float2(0.0f, 1.0f)));
		normalize(kernel[i]);

		//randomize points distance to sphere center, making them more concentrated towards the center
		kernel[i] *= nrand(float2(0.0f, 1.0f)); 
		float scale = float(i) / float(kernelSize);
		scale = lerp(0.1f, 1.0f, scale * scale);
		kernel[i] *= scale; 
	}

	float3 rvec = float3(rand(coord), rand(coord).x);
	float3 tangent = normalize(rvec - normal * dot(rvec, normal));
	float3 bitangent = cross(normal, tangent);

	//reorient our sample kernel along the origin's normal
	float3x3 tbn = float3x3(tangent, bitangent, normal);

	// calculate occlusion
	float occlusion = 0.0;
	for (i = 0; i < kernelSize; ++i) {
		// get sample positions around origin:
		float3 samplePoint = mul(kernel[i], tbn) * uRadius;
		samplePoint += origin;
		
		// project sample position:
		float3 screenSpaceSample = projectPosition (samplePoint);

		// get depth from depthbuffer at sample position:
		float sampleDepth = readDepth(screenSpaceSample.xy);
		
		// get sample point actual depth
		float actualDepth = samplePoint.z;

		// range check & accumulate:
		float rangeCheck= abs(actualDepth - sampleDepth) < uRadius ? 1.0 : 0.0;
		occlusion += (sampleDepth < actualDepth ? 1.0 : 0.0) * rangeCheck;
	}

	occlusion = 1.0 - (occlusion / kernelSize);

	return float4(occlusion, occlusion, occlusion, 1.0);
}




 
float4 AOPass(VSOUT IN) : COLOR0
{
	float depth = readDepth(IN.UVCoord);
	float3 pos = getPosition(IN.UVCoord);
	float depthPos = depth * depthRange;
	float dx = normalize(ddx(pos)).z * g_InvFocalLen.x;
	float dy = normalize(ddy(pos)).z * g_InvFocalLen.y;
	float cosdx = dx * cosAngleBias;
	float cosdy = dy * cosAngleBias;
	float sindx = sqrt(1 - dx * dx) * sinAngleBias;
	float sindy = sqrt(1 - dy * dy) * sinAngleBias;
	float d;
	float depth_pred;
	float ao = 0.0;
	float s = 0.0;
 
	if (depth >= 1) {
		ao = 1;
	}
	else {
		float2 noise = rand(IN.UVCoord);
		float w = (noise.x * (1.0 - noise.x)) / (g_InvFocalLen.x * depthPos / (TESR_ReciprocalResolution.x * 200000));
		float h = (noise.y * (1.0 - noise.y)) / (g_InvFocalLen.y * depthPos / (TESR_ReciprocalResolution.y * 200000)); 
		float pw;
		float ph;

		for (int i = 0; i < rings; i++)
		{
			[unroll]
			for (int j = 0; j < samples * i; j++)
			{
				float step = PI * 2.0 / float(samples * i);
				pw = (cos(float(j) * step) * float(i)) * TESR_AmbientOcclusionAOData.x;
				ph = (sin(float(j) * step) * float(i)) * TESR_AmbientOcclusionAOData.x * TESR_ReciprocalResolution.z;
				float new_x = IN.UVCoord.x + pw * w;
				float new_y = IN.UVCoord.y + ph * h;
				float anglebias_dx = cosdx + sign(pw) * sindx;
				float anglebias_dy = cosdy + sign(ph) * sindy;
				depth_pred = depth + depth * anglebias_dx * pw * w + depth * anglebias_dy * ph * h;
				d = readDepth(float2(new_x, new_y));
				float aoresult = compareDepths(depth_pred, d);
				ao += aoresult;
				s += 1;
			}
		}
		ao /= s;
		if (TESR_FogData.y > TESR_FogData.x && depthPos >= TESR_FogData.x) ao *= saturate(lerp(1, 0, (depthPos - TESR_FogData.x) / (TESR_FogData.y - TESR_FogData.x)));
		ao = 1.0 - ao * TESR_AmbientOcclusionAOData.y;
	}
	
	return float4(clamp(ao * ao, TESR_AmbientOcclusionAOData.z, 1), packDepth(depth), 1);
}

float4 BlurPS(VSOUT IN, uniform float2 OffsetMask) : COLOR0
{
	float WeightSum = 0.114725602f;
	float4 ao = tex2D(TESR_RenderedBuffer, IN.UVCoord);
	ao.r = ao.r * WeightSum;
 
	float Depth1 = unpackDepth(ao.gb);
	clip(0.9999 - Depth1);
 
	int i = 0;
    for (i = 0; i < cKernelSize; i++)
    {
		float2 uvOff = (BlurOffsets[i] * OffsetMask) * TESR_AmbientOcclusionData.w;
		float4 Color = tex2D(TESR_RenderedBuffer, IN.UVCoord + uvOff);
		float Depth2 = unpackDepth(Color.gb);
		float diff = abs(float(Depth1 - Depth2) * depthRange);
 
		if(diff <= TESR_AmbientOcclusionData.z)
		{
			ao.r += BlurWeights[i] * Color.r;
			WeightSum += BlurWeights[i];
		}
    }
	ao.r /= WeightSum;
    return ao;
}
 
float4 Combine(VSOUT IN) : COLOR0
{
	float3 color = tex2D(TESR_SourceBuffer, IN.UVCoord).rgb;
	float ao = tex2D(TESR_RenderedBuffer, IN.UVCoord).r;
	float luminance = color.r * 0.3 + color.g * 0.59 + color.b * 0.11;
	float white = 1.0;
	float black = 0.0;
	float lt = luminance - TESR_AmbientOcclusionData.y;
	luminance = clamp(max(black, lt) + max(black, lt) + max(black, lt), 0.0, 1.0);
	ao = lerp(ao, white, luminance);
	color *= ao;
	
    #if viewao
		return float4(ao, ao, ao, 1);
	#endif
	
	return float4(color, 1.0f);
   
}
 
technique
{
	pass
	{
		VertexShader = compile vs_3_0 FrameVS();
		PixelShader = compile ps_3_0 SSAO();
	}
	// pass
	// {
	// 	VertexShader = compile vs_3_0 FrameVS();
	// 	PixelShader = compile ps_3_0 AOPass();
	// }
	
	// pass
	// { 
	// 	VertexShader = compile vs_3_0 FrameVS();
	// 	PixelShader = compile ps_3_0 BlurPS(OffsetMaskH);
	// }
	
	// pass
	// { 
	// 	VertexShader = compile vs_3_0 FrameVS();
	// 	PixelShader = compile ps_3_0 BlurPS(OffsetMaskV);
	// }
	
	pass
	{
		VertexShader = compile vs_3_0 FrameVS();
		PixelShader = compile ps_3_0 Combine();
	}
}