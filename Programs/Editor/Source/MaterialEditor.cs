using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

using SpockEngine;
using Math = SpockEngine.Math;

#if false
struct MaterialInputs {
    vec4  baseColor;
    float roughness;
    float metallic;
    float reflectance;
    float ambientOcclusion;
    vec4  emissive;

    vec3 sheenColor;
    float sheenRoughness;

    float clearCoat;
    float clearCoatRoughness;

    float anisotropy;
    vec3  anisotropyDirection;

    float thickness;
    float subsurfacePower;
    vec3  subsurfaceColor;
    vec3  sheenColor;
    vec3  subsurfaceColor;

    vec3  normal;
    vec3  bentNormal;
    vec3  clearCoatNormal;
    vec4  postLightingColor;

    vec3 absorption;
    float transmission;
    float ior;
    float microThickness;
};
#endif

namespace SEEditor
{
    public class UITextureWithPreview : UIBoxLayout
    {
        UIImage mTexturePreview = new UIImage();
        public UITextureWithPreview(string aTitle) : base(eBoxLayoutOrientation.HORIZONTAL)
        {
            var lInfoLayout = new UIBoxLayout(eBoxLayoutOrientation.VERTICAL);
            var lLabel = new UILabel(aTitle);
            lLabel.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            lInfoLayout.Add(lLabel, 20.0f, false, true);

            // var lFilter = new UIComboBox();
            // lFilter.SetItemList(new string[] { "Linear", "Nearest" });
            // lInfoLayout.Add(lFilter, 30.0f, false, true);
            // var lMipFilter = new UIComboBox();
            // lMipFilter.SetItemList(new string[] { "Linear", "Nearest" });
            // lInfoLayout.Add(lMipFilter, 30.0f, false, true);
            // var lWrapping = new UIComboBox();
            // lWrapping.SetItemList(new string[] { "Repeat", "Mirrored repeat", "Clamp to edge", "Clamp to border", "Mirrored clamp to border" });
            // lInfoLayout.Add(lWrapping, 30.0f, false, true);

            Add(lInfoLayout, true, true);

            Add(mTexturePreview, 75, false, true);
        }
    }

    public class UIMaterialEditor : UIForm
    {
        UIBoxLayout mMainLayout = new UIBoxLayout(eBoxLayoutOrientation.VERTICAL);
        UILabel mMaterialName = new UILabel("MATERIAL_0");

        UILabel mShadingModelLabel = new UILabel("Shading:");
        string[] mShadingModels = new string[] { "Standart", "Subsurface", "Cloth", "Unlit" };
        UIComboBox mShadingModel = new UIComboBox();

        public UIMaterialEditor() : base()
        {
            SetTitle("EDIT MATERIAL");
            SetPadding(5.0f, 15.0f);


            mMainLayout.Add(mMaterialName, 30.0f, false, true);

            mShadingModelLabel.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);

            var lLayout0 = new UIBoxLayout(eBoxLayoutOrientation.HORIZONTAL);
            mShadingModel.SetItemList(mShadingModels);
            mShadingModel.CurrentItem = 1;
            lLayout0.Add(mShadingModelLabel, 100.0f, false, true);
            lLayout0.Add(mShadingModel, true, true);
            mMainLayout.Add(lLayout0, 30.0f, false, true);


            var lLabel10 = new UILabel("Line width");
            mMainLayout.Add(lLabel10, 30.0f, false, true);

            var lLabel11 = new UILabel("Is two sided");
            mMainLayout.Add(lLabel11, 30.0f, false, true);

            var lLabel12 = new UILabel("Culling");
            mMainLayout.Add(lLabel12, 30.0f, false, true);

            var lLabel14 = new UILabel("Use Alpha mask");
            mMainLayout.Add(lLabel14, 30.0f, false, true);

            var lLabel15 = new UILabel("Alpha mask threshold");
            mMainLayout.Add(lLabel15, 30.0f, false, true);

            var lLabel16 = new UILabel("Blend mode");
            mMainLayout.Add(lLabel16, 30.0f, false, true);

            var lTextureHeight = 75.0f;

            var lLabel0 = new UILabel("BASIC PROPERTIES");
            // lLabel0.SetFont(eFontFamily.H2);
            lLabel0.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            mMainLayout.Add(lLabel0, 30.0f, false, true);
            var lAlbedoTexture = new UITextureWithPreview("Albedo");
            mMainLayout.Add(lAlbedoTexture, lTextureHeight, false, true);
            var lNormalsTexture = new UITextureWithPreview("Normals");
            mMainLayout.Add(lNormalsTexture, lTextureHeight, false, true);
            var lMetalRoughTexture = new UITextureWithPreview("Metal/Rough");
            mMainLayout.Add(lMetalRoughTexture, lTextureHeight, false, true);
            var lOcclusionTexture = new UITextureWithPreview("Occlusion");
            mMainLayout.Add(lOcclusionTexture, lTextureHeight, false, true);
            var lOcclusionStrengthTexture = new UITextureWithPreview("Occlusion strength");
            mMainLayout.Add(lOcclusionStrengthTexture, lTextureHeight, false, true);
            var lEmissiveTexture = new UITextureWithPreview("Emissive");
            mMainLayout.Add(lEmissiveTexture, lTextureHeight, false, true);

            var lLabel1 = new UILabel("SUBSURFACE PROPERTIES");
            // lLabel1.SetFont(eFontFamily.H2);
            lLabel1.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            mMainLayout.Add(lLabel1, 30.0f, false, true);
            var lSubsurfacePowerTexture = new UITextureWithPreview("Subsurface power");
            mMainLayout.Add(lSubsurfacePowerTexture, lTextureHeight, false, true);
            var lSubsurfaceColorTexture = new UITextureWithPreview("Subsurface color");
            mMainLayout.Add(lSubsurfaceColorTexture, lTextureHeight, false, true);

            var lLabel4 = new UILabel("CLOTH PROPERTIES");
            // lLabel4.SetFont(eFontFamily.H2);
            lLabel4.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            mMainLayout.Add(lLabel4, 30.0f, false, true);
            var lSheenTexture = new UITextureWithPreview("Sheen color");
            mMainLayout.Add(lSheenTexture, lTextureHeight, false, true);
            var lSheenRoughnessTexture = new UITextureWithPreview("Sheen roughness");
            mMainLayout.Add(lSheenRoughnessTexture, lTextureHeight, false, true);

            var lLabel2 = new UILabel("CLEAR COAT");
            // lLabel2.SetFont(eFontFamily.H2);
            lLabel2.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            mMainLayout.Add(lLabel2, 30.0f, false, true);
            var lColorTexture = new UITextureWithPreview("Color");
            mMainLayout.Add(lColorTexture, lTextureHeight, false, true);
            var lThicknessTexture = new UITextureWithPreview("Thickness");
            mMainLayout.Add(lThicknessTexture, lTextureHeight, false, true);
            var lClearCoatRoughnessTexture = new UITextureWithPreview("Roughness");
            mMainLayout.Add(lClearCoatRoughnessTexture, lTextureHeight, false, true);
            var lClearCoatNormalTexture = new UITextureWithPreview("Normals");
            mMainLayout.Add(lClearCoatNormalTexture, lTextureHeight, false, true);
            var lBentNormalTexture = new UITextureWithPreview("Bent normals");
            mMainLayout.Add(lBentNormalTexture, lTextureHeight, false, true);

            var lLabel3 = new UILabel("ANISOTROPY");
            // lLabel3.SetFont(eFontFamily.H2);
            lLabel3.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            mMainLayout.Add(lLabel3, 30.0f, false, true);
            var lAnisotropyTexture = new UITextureWithPreview("Anisotropy");
            mMainLayout.Add(lAnisotropyTexture, lTextureHeight, false, true);
            var lAnisotropyNormalsTexture = new UITextureWithPreview("Normals");
            mMainLayout.Add(lAnisotropyNormalsTexture, lTextureHeight, false, true);


            var lLabel30 = new UILabel("OTHER");
            // lLabel3.SetFont(eFontFamily.H2);
            lLabel30.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
            mMainLayout.Add(lLabel30, 30.0f, false, true);
            var lAbsorptionTexture = new UITextureWithPreview("Absorption");
            mMainLayout.Add(lAbsorptionTexture, lTextureHeight, false, true);
            var lTransmissionTexture = new UITextureWithPreview("Transmission");
            mMainLayout.Add(lTransmissionTexture, lTextureHeight, false, true);
            var lIorTexture = new UITextureWithPreview("IOR");
            mMainLayout.Add(lIorTexture, lTextureHeight, false, true);
            var lReflectanceTexture = new UITextureWithPreview("Reflectance");
            mMainLayout.Add(lReflectanceTexture, lTextureHeight, false, true);
            var lMicroThicknessTexture = new UITextureWithPreview("Micro thickness");
            mMainLayout.Add(lMicroThicknessTexture, lTextureHeight, false, true);

            SetContent(mMainLayout);
        }

        public void Update()
        {
            base.Update();
        }
    }
}
