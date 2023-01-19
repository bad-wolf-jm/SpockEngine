using System;

namespace SpockEngine
{
    public class UI
    {
        public static void Text(string aText)
        {
            CppCall.UI_Text(aText);
        }
    };
}