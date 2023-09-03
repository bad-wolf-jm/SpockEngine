using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UICheckBoxWithLabel : UIBoxLayout
    {
        UICheckBox mCheckBox = new UICheckBox();
        UILabel mLabel = new UILabel();

        public bool IsChecked
        {
            get { return mCheckBox.IsChecked; }
            set { mCheckBox.IsChecked = value; }
        }

        public void OnClick(UICheckBox.OnClickDelegate x)
        {
            mCheckBox.OnClick(x);
        }

        public UICheckBoxWithLabel(string aLabel) : base(eBoxLayoutOrientation.HORIZONTAL)
        {
            SetItemSpacing(1.0f);
            Add(mCheckBox, false, true);
            Add(mLabel, true, true);

            mLabel.SetText(aLabel);
            mLabel.SetAlignment(eHorizontalAlignment.LEFT, eVerticalAlignment.CENTER);
        }
    }
}
