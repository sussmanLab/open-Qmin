#include "DatabaseNetCDF.h"
/*! \file DatabaseNetCDF.cpp */

BaseDatabaseNetCDF::BaseDatabaseNetCDF(string fn, NcFile::FileMode mode)
     : baseDatabase(fn,mode),
     File(fn.c_str(), mode)
{
    NcError err(NcError::silent_nonfatal);
}


