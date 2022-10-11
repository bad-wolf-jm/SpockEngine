function [baseline, noise_std, file_info] = stats(data, cfg, prec_incr=1)
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Trace statistics
%
% [baseline, noise_std] = stats(data, cfg)
%
% data : Trace data samples
% cfg  : configuration structure
%
% baseline : Noise baseline
% noise_std: Noise standard deviation
% file_info: File information for results writing
%

  PREC_INCR = 2*(ceil(prec_incr/2));
	win_len = 2^cfg.cfg.win_size;

  if( win_len > numel(data) )
  	error('Number of samples is too low for statistics configuration')
  end


% convert to column vector
	sz = size(data);
	if sz(2) > sz(1)
	  d = data';
	else
	  d = data;
	endif

  s = sum(d(1:win_len));
  baseline = roundb(s/win_len);
  
  sum_sq = sum(d(1:win_len).^2);
  sq_sum = sum(d(1:win_len))^2;
  
  var_part = abs(sum_sq*win_len - sq_sum); % work on sum; keep maximum precision
  
  noise_std =  roundb(floor(sqrt(var_part*2^PREC_INCR))/(2^(PREC_INCR/2)*win_len));
  
  file_info.header = { 'trace', 'baseline', 'noise_std' };
  file_info.data = [ baseline, noise_std ];

endfunction
% vim:tw=0:ts=2:sts=2:sw=2:et