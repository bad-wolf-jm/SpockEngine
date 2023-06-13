using System;

namespace SpockEngine
{
   public class UIComboBoxWithLabel : UIBoxLayout
   {
      private string[] mStrings = new string[] { };
      private UILabel mLabel = new UILabel("");
      private UIComboBox mCombo = new UIComboBox();

      public UIComboBoxWithLabel() : base(eBoxLayoutOrientation.HORIZONTAL)
      {
         Add(mLabel, 75.0f, true, true);
         Add(mCombo, true, true);
      }

      public string Label { set { mLabel.SetText(value); } }

      private string[] mValues;
      public string[] Values
      {
         get { return mValues; }
         set { mValues = value; mCombo.SetItemList(mValues); }
      }

      public int CurrentItem
      {
         get { return mCombo.CurrentItem; }
         set { mCombo.CurrentItem = value; }
      }

      public void OnChanged(UIComboBox.ChangedDelegate aHandler)
      {
         mCombo.OnChanged(aHandler);
      }
   }
} // namespace SE.OtdrEditor