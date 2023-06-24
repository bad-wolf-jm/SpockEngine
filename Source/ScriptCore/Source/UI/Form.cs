using System;
using System.Runtime;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIForm : UIComponent
    {
        public UIForm() : this(Interop.UIForm_Create()) { }

        public UIForm(IntPtr aDerived) : base(aDerived) { }

        ~UIForm() { Interop.UIForm_Destroy(mInstance); }

        public void SetTitle(string aTitle) { Interop.UIForm_SetTitle(mInstance, aTitle); }

        UIComponent mContent;
        public void SetContent(UIComponent aContent) { mContent = aContent; Interop.UIForm_SetContent(mInstance, aContent.Instance); }
        public void SetSize(float aWidth, float aHeight) { Interop.UIForm_SetSize(mInstance, aWidth, aHeight); }

        public void Update() { Interop.UIForm_Update(mInstance); }
    }
}
