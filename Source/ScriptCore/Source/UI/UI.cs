using System;

namespace SpockEngine
{
    public class UI
    {
        public static void Text(string aText)
        {
            CppCall.UI_Text(aText);
        }
 
        public static bool Button(string aText)
        {
            return CppCall.UI_Button(aText);
        }
    };
}