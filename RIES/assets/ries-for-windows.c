/* ries-for-windows.c

    Wrapper source file for compiling RIES in Microsoft Visual C (or C++)
    Copyright (C) 2000-2018 Robert P. Munafo
    This is the 2018 Aug 12 version of "ries-for-windows.c"


    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

    If you got ries.c from the website www.mrob.com, the
    GNU General Public License may be retrieved from the following URL:

    http://www.mrob.com/ries/COPYING.txt
*/

/* Instructions:

(This is based partly on msdn.microsoft.com/en-us/library/ms235629.aspx
and I may have gotten parts wrong)

1. Go to the RIES website (mrob.com/ries) and follow the "Source code"
   link. Download and save the source files. There should be:

    * This file, called "ries-for-windows.c"
    * The main source code, called "ries.c"
    * Optionally, the stand-alone maths functions library "msal_math64.c"

2. Make a new folder (directory) and put the downloaded files there.

3. In Visual Studio, select File > New > Project from Existing Code
   Select "Visual C++"
   Project file location: select the directory you created in step 2.
   Project name: give the project any name you wish.
   Project type: select "Win32", then "Win32 Console Application"
   You may leave "precompiled header" selected.

4. In Solution Explorer, right-click "Source files" and select Add >
   Existing Item... Select the source file "ries-for-windows.c".
   Do not add the other files like "ries.c". (they will be #included by
   ries-for-windows.c).

5. It should compile and run (using the green "play" button). If not,
   comment out the line that says #include "stdafx.h", and then go into
   Configuration Properties -> C/C++ -> Precompiled Headers
   and select Precompiled Header: Not Using Precompiled Headers

6. RIES will run a lot faster with these additional project
   configuration settings:

   Configuration Properties -> C/C++ -> Code Generation
     Basic Runtime Checks: Default
     Enable Minimal Rebuild: No (/Gm-)
   Configuration Properties -> C/C++ -> General
     Debug Information Format: C7 compatible (/Z7)
   Configuration Properties -> Linker -> General
     Enable Incremental Linking: No (/INCREMENTAL:NO)
   Configuration Properties -> Linker -> Optimization
     Link-Time Code Generation: Use Link Time Code Generation (/LTCG)
   Configuration Properties -> C/C++ -> Optimization
     Optimization: Full Optimization (/Ox)
     Enable Intrinsic Functions: Yes (/Oi)
     Favor Size Or Speed: Favor fast code (/Ot)
     Omit Frame Pointers: Yes (/Oy)
     Enable Fiber-Safe Optimizations: Yes (/GT)
     Whole Program Optimization: Yes (/GL)

7. Again, compile and run (with the green "play" button). You'll probably see
   "this project is out of date, do you want to build it?", select Yes.
   A command window should appear briefly, displaying a message like
   "ries: Please specify a target number." and some lines of instructions.

8. Use Debug > Build Solution to build an executable. It will put it in
   "Debug\ries.exe", or something different depending on which build
   configuration you currently have active and what what you chose as the
   project name. The EXE can be run from the command line with arguments.

If you turned off the precompiled headers option in the VC++ project,
you can compile the "ries.c" source directly, rather than using this
stub. With precompiled headers turned off RIES will compile more
slowly. If you aren't compiling RIES often, this may be preferable.

In either case, keep in mind that RIES is vanilla C with stdio: it
interacts with the user only through the console window (command-line
interface).

*/

#include "stdafx.h"

#define RIES_USED_RFWC
#include "ries.c"

