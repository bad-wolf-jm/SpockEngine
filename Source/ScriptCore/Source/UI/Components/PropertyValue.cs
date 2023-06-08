using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIPropertyValue : UIBoxLayout
    {
        public UIPropertyValue() : base(UIPropertyValue_Create(), true) { }

        public UIPropertyValue(string aText) : base(UIPropertyValue_CreateWithText(aText), true) { }

        public UIPropertyValue(string aText, eBoxLayoutOrientation aOrientation) : base(UIPropertyValue_CreateWithTextAndOrientation(aText, aOrientation), true) { }

        ~UIPropertyValue() { UIPropertyValue_Destroy(mInstance); }

        public void SetValue(string aText) { UIPropertyValue_SetValue(mInstance, aText); }

        public void SetValueFont(eFontFamily aFont) { UIPropertyValue_SetValueFont(mInstance, aFont); }

        public void SetNameFont(eFontFamily aFont) { UIPropertyValue_SetNameFont(mInstance, aFont); }
    }
}
