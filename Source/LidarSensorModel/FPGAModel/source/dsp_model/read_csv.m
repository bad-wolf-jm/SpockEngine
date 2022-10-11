function d = read_csv(filename, separator=',')
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Read .csv file 
%
% d = read_csv(filename)
%
% filename : Filename of input data
% separator: Field separator character
%
% d: multi-dimensional data array
%
  pkg load io

  a = cell2mat(csv2cell(filename,2));

  for i=1:rows(a)
     d{i} = a(i,2:end);
  end; 

endfunction
