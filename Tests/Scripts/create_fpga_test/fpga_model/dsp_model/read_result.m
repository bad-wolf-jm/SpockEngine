function d = read_result(filename, separator=';')
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Read result_*.csv file
%
% d = read_result(filename)
%
% filename : Filename of input data
% separator: Field separator character
%
% d: multidimensional array of structure
% beware; this is not a structure of arrays and must be accesses as:
%  d(:).field; therefore, to use the array, one must use:
% [d.field] which will return a 1d array and must be reshaped; thus
% reshape([d.field], [], columns(d))
%
  pkg load io

  c = csv2cell(filename,separator);
  s = cell2struct(c(2:end,:), c(1,:), 2);
  n = numel(find([s.trace]==0));
  s1 = s(1:floor(numel(s)/n)*n);
  d = reshape(s1, n, []);
endfunction
