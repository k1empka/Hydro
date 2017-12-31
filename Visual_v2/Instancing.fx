//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
texture g_txScene : TEXTURE;
float4x4 g_mWorld : WORLD : register(c0);
float4x4 g_mView : VIEW : register(c4);
float4x4 g_mProj : PROJECTION : register(c8);

float4 g_BoxInstance_Position : BOXINSTANCE_POSITION : register(c13);
float4 g_BoxInstance_Color : BOXINSTANCE_COLOR : register(c14);

#define g_nNumBatchInstance 120
float4 g_vBoxInstance_Position[g_nNumBatchInstance] : BOXINSTANCEARRAY_POSITION : register(c16);
float4 g_vBoxInstance_Color[g_nNumBatchInstance] : BOXINSTANCEARRAY_COLOR : register(c136);

float alpha = 1;

//-----------------------------------------------------------------------------
// Texture samplers
//-----------------------------------------------------------------------------
sampler g_samScene =
sampler_state
{
    Texture = <g_txScene>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = Linear;
    //BlendEnable[0] = TRUE;
    //CullMode = None;
    //AlphaBlendEnable = TRUE;
    //DestBlend = INV_SRC_ALPHA;
    //SrcBlend = SRC_ALPHA;
};

void VS_ShaderInstancing(float4 vPos : POSITION,
						float3 vNormal : NORMAL,
						float2 vTex0 : TEXCOORD0,
						float vBoxInstanceIndex : TEXCOORD1,
						out float4 oPos : POSITION,
						out float4 oColor : COLOR0,
						out float2 oTex0 : TEXCOORD0)
{
    float4 vBoxInstance = g_vBoxInstance_Position[vBoxInstanceIndex];
    vBoxInstance.w *= 10 * 3.1415;
    float4 vRotatedPos = vPos;
    vRotatedPos.x = vPos.x * cos(vBoxInstance.w) + vPos.z * sin(vBoxInstance.w);
    vRotatedPos.z = vPos.z * cos(vBoxInstance.w) - vPos.x * sin(vBoxInstance.w);

    vRotatedPos += float4(vBoxInstance.xyz * 32 - 16, 0); // Camera

    oPos = mul(vRotatedPos, g_mWorld);
    oPos = mul(oPos, g_mView);
    oPos = mul(oPos, g_mProj);

    oTex0 = vTex0;
    oColor = g_vBoxInstance_Color[vBoxInstanceIndex];
}

float4 PS(float2 vTex0 : TEXCOORD0,
			float4 vColor : COLOR0) : COLOR0
{
    return tex2D( g_samScene, vTex0 ) * vColor;
    //float4 outColor = tex2D(g_samScene, vTex0) * vColor;
    //return outColor;
}

technique TShader_Instancing
{
    pass P0
    {
        ZEnable = true;
        //FillMode = Wireframe;
        //CullMode = None;
        //AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;
        
        VertexShader = compile vs_2_0 VS_ShaderInstancing();
        PixelShader = compile ps_2_0 PS();
    }
}