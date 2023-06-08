using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIForm : UIComponent
    {
        public UIForm() : this(UIForm_Create()) { }

        public UIForm(ulong aDerived) : base(aDerived) { }

        ~UIForm() { UIForm_Destroy(mInstance); }

        public void SetTitle(string aTitle) { UIForm_SetTitle(mInstance, aTitle); }

        public void SetContent(UIComponent aContent) { UIForm_SetContent(mInstance, aContent.Instance); }
        public void SetSize(float aWidth, float aHeight) { UIForm_SetSize(mInstance, aWidth, aHeight); }

        public void Update() { UIForm_Update(mInstance); }
    }
}
