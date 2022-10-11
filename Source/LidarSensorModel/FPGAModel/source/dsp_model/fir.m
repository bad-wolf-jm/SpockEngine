function [df, aux_d, file_info] = fir(data, cfg, aux=[])
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Low pass filter
%
% [ df, aux_d } = stats(data, cfg, aux)
%
% data : Trace data samples
% cfg  : Configuration structure
% aux  : Auxuliary stream to delay
%
% df    : Filtered data
% aux_d : Delayed auxiliary stream
% file_info: File information for results writing
%

  if( numel(cfg.coefficients) >= numel(data) )
    error('Not enough data to filter with such coefficients')
  endif

% convert to column vector
  sz = size(data);
  if sz(2) > sz(1)
    d = data';
    nb_traces = sz(1);
  else
    d = data;
    nb_traces = sz(2);
  endif

  sz = size(aux);
  if sz(2) > sz(1)
    a = aux';
  else
    a = aux;
  endif


  if( !cfg.block_enable )
    df = d;
    aux_d = a;
  else

    % trailing 0s are not poart of the delay
    delay = floor(numel([1:find(cfg.coefficients, 1, 'last')])/2); % delay is filter order /2; filter order if nb coefs-1

    if( cfg.ctrl.delay_sideinfo~=delay )
      fprintf('FIR: Calculated delay (%u) is not equal to provided delay (%u);\n', delay, cfg.ctrl.delay_sideinfo);
    endif;

    % filter with precharge
    nb_coefs = numel(cfg.coefficients);
    df = roundb(conv2(cfg.coefficients ./ 2^15, 1, [ones(nb_coefs-1, nb_traces).*d(1,:);d]))(nb_coefs:end-(nb_coefs-1),:);

    if( ~isempty(aux) && numel(aux)==numel(d) )
      aux_d = [zeros(cfg.ctrl.delay_sideinfo, columns(aux));aux(1:end-cfg.ctrl.delay_sideinfo,:)];
    else
      aux_d = zeros(rows(data), nb_traces);
    endif

  endif;


  file_info.header = { 'trace'; 'data_o'; 'blind_on'; 'last' };
  file_info.data = reshape([df;aux_d], size(df,1), []); % zip

endfunction
% vim:tw=0:ts=2:sts=2:sw=2:et