function d = read_dat(filename, separator=' ', trace_num=0)
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Read .dat file generated for simulation
%
% d = read_dat(filename)
%
% filename : Filename of input data
% separator: Field separator character
% trace_num: Optionaly indicate a single trace to be read (0 = all) 
%
% d: multi-dimensional data array
%

  f = fopen(filename,'r');

  % get file header to acquire structure
  head = fgetl(f);
  h = regexp(head, separator, 'split');

  % read data from separated fields
  data = textscan(f, strjoin(repmat({'%f'}, [1, numel(h)]), separator), 'HeaderLines', 1);
  fclose(f);

  pStart = find(diff([0;data{3}])==1);
  pEnd = find(diff([data{3};0])==-1);


  if (trace_num == 0)
    for i=1:length(pStart)
       d{i} = data{2}(pStart(i):pEnd(i));
    end
  else 
     d = data{2}(pStart(trace_num):pEnd(trace_num));
  endif;  

endfunction
