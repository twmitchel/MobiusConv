/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef CMD_LINE_PARSER_INCLUDED
#define CMD_LINE_PARSER_INCLUDED
#include <stdarg.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef WIN32
int strcasecmp(char* c1,char* c2);
#endif

class cmdLineReadable{
public:
	bool set;
	char *name;
    char *description;
    char shortName;
	cmdLineReadable(const char *name, char shortName = -1);
    void setDescription(const char *desc);
    void printDescription();

	virtual ~cmdLineReadable(void);
	virtual int read(char** argv,int argc);
	virtual void writeValue(char* str);
    virtual bool expectsArg() { return false; }
};

class cmdLineInt : public cmdLineReadable {
public:
	int value;
	cmdLineInt(const char *name, char shortName = -1);
	cmdLineInt(const char *name, const int& v, char shortName = -1);
	int read(char** argv,int argc);
	void writeValue(char* str);

    bool expectsArg() { return true; }
};

class cmdLineIntSequence : public cmdLineReadable
{
public:
	/** Define the sequence */
	int start, increment, end;
	/** Used for iteration */ 
	int value;
	cmdLineIntSequence(const char *name, char shortName = -1);
	cmdLineIntSequence(const char *name, const int v, char shortName = -1);
	int read(char** argv,int argc);
	void writeValue(char* str);
	bool expectsArg() { return true; }

	void reset()
	{
		value = start;
	}

	bool advance()
	{
		if (value + increment > end)
			return false;
		value += increment;
		return true;
	}
};

template<int Dim>
class cmdLineIntArray : public cmdLineReadable {
public:
	int values[Dim];
	cmdLineIntArray(const char *name, char shortName = -1);
	cmdLineIntArray(const char *name, const int v[Dim], char shortName = -1);
	int read(char** argv,int argc);
	void writeValue(char* str);
    bool expectsArg() { return true; }
};
class cmdLineInts : public cmdLineReadable
{
public:
	int count;
	int* values;
	cmdLineInts( const char* name , char shortName=-1 );
	~cmdLineInts( void );
	int read( char** argv , int argc );
	void writeValue( char* str );
    bool expectsArg( void ) { return true; }
};

class cmdLineFloat : public cmdLineReadable {
public:
	float value;
	cmdLineFloat(const char *name, char shortName = -1);
	cmdLineFloat(const char *name, const float& f, char shortName = -1);
	int read(char** argv,int argc);
	void writeValue(char* str);
    bool expectsArg() { return true; }
};
template<int Dim>
class cmdLineFloatArray : public cmdLineReadable {
public:
	float values[Dim];
	cmdLineFloatArray(const char *name, char shortName = -1);
	cmdLineFloatArray(const char *name, const float f[Dim], char shortName = -1);
	int read(char** argv,int argc);
	void writeValue(char* str);
    bool expectsArg() { return true; }
};
class cmdLineString : public cmdLineReadable {
public:
	char* value;
	cmdLineString(const char *name, char shortName = -1);
	~cmdLineString();
	int read(char** argv,int argc);
	void writeValue(char* str);
    bool expectsArg() { return true; }
};
class cmdLineStrings : public cmdLineReadable {
public:
	int count;
	char** values;
	cmdLineStrings(const char *name, char shortName = -1);
	~cmdLineStrings(void);
	int read(char** argv,int argc);
	void writeValue(char* str);
    bool expectsArg() { return true; }
};
template<int Dim>
class cmdLineStringArray : public cmdLineReadable {
public:
	char* values[Dim];
	cmdLineStringArray(const char *name, char shortName = -1);
	~cmdLineStringArray();
	int read(char** argv,int argc);
	void writeValue(char* str);
    bool expectsArg() { return true; }
};

void cmdLineParse(int argc, char **argv, cmdLineReadable** params,
        std::vector<std::string> &nonoptArgs, int *argcStripped = (int *) NULL,
        char ***argvStripped = (char ***) NULL);

char* FileExtension( char* fileName );
char* LocalFileName( char* fileName );
char* DirectoryName( char* fileName );
char* GetFileExtension( char* fileName );
char* GetLocalFileName( char* fileName );
char** ReadWords( const char* fileName , int& cnt );

#include "CmdLineParser.inl"
#endif // CMD_LINE_PARSER_INCLUDED
