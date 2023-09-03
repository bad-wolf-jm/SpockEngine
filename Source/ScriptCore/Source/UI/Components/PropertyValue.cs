using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIPropertyValue : UIBoxLayout
    {
        public UIPropertyValue() : base(Interop.UIPropertyValue_Create(), true) { }

        public UIPropertyValue(string aText) : this()
        {
            SetText(aText);
            SetOrientation(eBoxLayoutOrientation.HORIZONTAL);
        }

        public UIPropertyValue(string aText, eBoxLayoutOrientation aOrientation) : this()
        {
            SetText(aText);
            SetOrientation(aOrientation);
        }

        ~UIPropertyValue() { Interop.UIPropertyValue_Destroy(mInstance); }

        public void SetText(string aText) { Interop.UIPropertyValue_SetText(mInstance, aText); }
        
        public void SetOrientation(eBoxLayoutOrientation aOrientation) { Interop.UIPropertyValue_SetOrientation(mInstance, aOrientation); }

        public void SetValue(string aText) { Interop.UIPropertyValue_SetValue(mInstance, aText); }

        public void SetValueFont(eFontFamily aFont) { Interop.UIPropertyValue_SetValueFont(mInstance, aFont); }

        public void SetNameFont(eFontFamily aFont) { Interop.UIPropertyValue_SetNameFont(mInstance, aFont); }
    }
}
