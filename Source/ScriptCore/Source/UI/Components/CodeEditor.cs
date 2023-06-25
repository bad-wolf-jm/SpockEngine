using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UICodeEditor : UIComponent
    {
        public struct Coordinates
        {
            int line, column;
        }

        private bool mDerived = false;

        public UICodeEditor() : this(Interop.UICodeEditor_Create(), false) { }
        public UICodeEditor(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }
        ~UICodeEditor() { if (!mDerived) Interop.UICodeEditor_Destroy(mInstance); }

        public void SetText(string aText) { Interop.UICodeEditor_SetText(mInstance, aText); }

        public void InsertText(string aText) { Interop.UICodeEditor_InsertText(mInstance, aText); }

        public string GetText() { return Interop.UICodeEditor_GetText(mInstance); }

        public string SelectedText { get { return Interop.UICodeEditor_GetSelectedText(mInstance); } }

        public string CurrentLineText { get { return Interop.UICodeEditor_GetCurrentLineText(mInstance); } }

        public bool ReadOnly
        {
            get { return Interop.UICodeEditor_GetReadOnly(mInstance); }
            set { Interop.UICodeEditor_SetReadOnly(mInstance, value); }
        }

        public Coordinates CursorPosition
        {
            get { return Interop.UICodeEditor_GetCursorPosition(mInstance); }
            set { Interop.UICodeEditor_SetCursorPosition(mInstance, value); }
        }

        public bool ShowWhitespace
        {
            get { return Interop.UICodeEditor_GetShowWhitespace(mInstance); }
            set { Interop.UICodeEditor_SetShowWhitespace(mInstance, value); }
        }

        public int TabSize
        {
            get { return Interop.UICodeEditor_GetTabSize(mInstance); }
            set { Interop.UICodeEditor_SetTabSize(mInstance, value); }
        }

        public void MoveUp(int aAmount, bool aSelect)
        {
            Interop.UICodeEditor_MoveUp(mInstance, aAmount, aSelect);
        }

        public void MoveDown(int aAmount, bool aSelect)
        {
            Interop.UICodeEditor_MoveDown(mInstance, aAmount, aSelect);
        }

        public void MoveLeft(int aAmount, bool aSelect, bool aWordMode)
        {
            Interop.UICodeEditor_MoveLeft(mInstance, aAmount, aSelect, aWordMode);
        }

        public void MoveRight(int aAmount, bool aSelect, bool aWordMode)
        {
            Interop.UICodeEditor_MoveRight(mInstance, aAmount, aSelect, aWordMode);
        }

        public void MoveTop(int aAmount, bool aSelect)
        {
            Interop.UICodeEditor_MoveTop(mInstance, aSelect);
        }

        public void MoveBottom(int aAmount, bool aSelect)
        {
            Interop.UICodeEditor_MoveBottom(mInstance, aSelect);
        }

        public void MoveHome(int aAmount, bool aSelect)
        {
            Interop.UICodeEditor_MoveHome(mInstance, aSelect);
        }

        public void MoveEnd(int aAmount, bool aSelect)
        {
            Interop.UICodeEditor_MoveEnd(mInstance, aSelect);
        }

        public void SetSelectionStart(Coordinates aPosition)
        {
            Interop.UICodeEditor_SetSelectionStart(mInstance, aPosition);
        }

        public void SetSelectionEnd(Coordinates aPosition)
        {
            Interop.UICodeEditor_SetSelectionEnd(mInstance, aPosition);
        }

        public void SetSelection(Coordinates aStart, Coordinates aEnd, int aMode)
        {
            Interop.UICodeEditor_SetSelection(mInstance, aStart, aEnd, aMode);
        }

        public void SelectWordUnderCursor()
        {
            Interop.UICodeEditor_SelectWordUnderCursor(mInstance);
        }

        public void SelectAll()
        {
            Interop.UICodeEditor_SelectAll(mInstance);
        }

        public bool HasSelection()
        {
            return Interop.UICodeEditor_HasSelection(mInstance);
        }

        public void Cut()
        {
            Interop.UICodeEditor_Cut(mInstance);
        }

        public void Copy()
        {
            Interop.UICodeEditor_Copy(mInstance);
        }

        public void Paste()
        {
            Interop.UICodeEditor_Paste(mInstance);
        }

        public void Delete()
        {
            Interop.UICodeEditor_Delete(mInstance);
        }

        public bool CanUndo()
        {
            return Interop.UICodeEditor_CanUndo(mInstance);
        }

        public bool CanRedo()
        {
            return Interop.UICodeEditor_CanRedo(mInstance);
        }

        public void Undo(int aSteps)
        {
            Interop.UICodeEditor_Undo(mInstance, aSteps);
        }

        public void Redo(int aSteps)
        {
            Interop.UICodeEditor_Redo(mInstance, aSteps);
        }


    }
}
