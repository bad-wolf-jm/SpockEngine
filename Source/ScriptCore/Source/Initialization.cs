using System;
using System.IO;

namespace SpockEngine
{

    public class IO
    {
        class ConsoleStream : Stream
        {
            public override bool CanRead { get { return false; } }
            public override bool CanSeek { get { return false; } }
            public override bool CanWrite { get { return true; } }
            public override long Length { get { return 0; } }
            public override long Position { get { return 0; } set { } }

            public override int Read(byte[] buffer, int offset, int count)
            {
                return 0;
            }

            public override long Seek(long offset, SeekOrigin origin)
            {
                return 0;
            }

            public override void Write(byte[] buffer, int offset, int count)
            {
                string result = System.Text.Encoding.UTF8.GetString(buffer, offset, count);

                CppCall.Console_Write(result);
            }

            public override void Flush()
            {
            }

            public override void SetLength(long value)
            {

            }
        }

        private static ConsoleStream mConsoleStream;

        public static void Initialize()
        {
            mConsoleStream = new ConsoleStream();
            var lConsoleStreamWriter = new StreamWriter(mConsoleStream, System.Text.Encoding.UTF8);

            Console.SetOut(lConsoleStreamWriter);
        }
    }

}