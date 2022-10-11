function [t, file_info] = dual_cfar_cago(data, cfg)
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Dual CFAR CAGO
%
% [t, file_info] = dual_cfar_cago(data, cfg)
%
% data : Incoming data
% cfg  : Configuration structure
%
% t         : Threshold
% file_info : Result file otuput structure
%


% convert to column vector
	sz = size(data);
	if sz(2) > sz(1)
	  d = data';
	else
	  d = data;
	endif

	for j=1:2
		[ cf(j).t, c(j).s ] = cfar(d, 2.^cfg.cfg.ref_length(j), cfg.cfg.guard_size(j), cfg.cfg.th_factor(j)/2^4, cfg.cfg.min_std(j) );
	endfor

	t = min(cf(1).t, cf(2).t);


  file_info.header = { 'trace', 'data_o', 'threshold', 'last' };
  file_info.data = [ d, t ];


endfunction