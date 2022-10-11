function [t, mu, sigma, file_info] = cfar(data, cfg, aux=[], clusters=2)
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Constant False Alarm Rate
%
% [t, mu, sigma] = cfar(data, cfg, clusters=2)
%
% data     : Trace data samples
% cfg      : Configuration structure
% clusters : Number of Window/Guard pairs per side
%
% t     : Threshold
% mu    : Windowed average
% sigma : Estimated standard deviation
% file_info: File information for results writing
%

%% internal functions
pkg load statistics

  function m = mink(A, k=0)
    a = A;
    [m, r] = min(a,[],2);
    for j=1:k
      a(sub2ind(size(a), (1:rows(a)), r')) = inf;
      [m, r] = min(a,[],2);
    endfor
  endfunction


  function m = maxk(A, k=0)
    a = A;
    [m, r] = max(a,[],2);
    for j=1:k
      a(sub2ind(size(a), (1:rows(a)), r')) = -inf;
      [m, r] = max(a,[],2);
    endfor
  endfunction


  win = 2.^cfg.cfg.ref_len;
  guard = cfg.cfg.guard_len;
  th_factor = cfg.cfg.th_factor/2^4;
  min_std = cfg.cfg.min_std;
  n_skip = cfg.cfg.n_skip;



%% processing

  if( clusters*(win+guard)+1 >= numel(data) )
    error('Number of samples is too low for CFAR configuration')
  end


% convert to column vector
  sz = size(data);
  if sz(2) > sz(1)
    d = data';
  else
    d = data;
  endif

%[u_l, wl, u_r, wr] = msum(d, win);


% get the means left and right
  idx = repmat([1:win],numel(d)-win+1,1).+[0:numel(d)-win]'; % shifting index
  D = d(idx); % each row is win data shifted by 1
  u = sum(D,2);
  s = sum(abs(D*win-u), 2); % still no division...


% build the multi window (taking into accound left and right shifts)
  u_w = [u;nan(win+guard, 1)];
  s_w = [s;nan(win+guard, 1)];
  for j=1:clusters-1,
    u_w = [[nan(win+guard, j);u_w],[u;nan((j+1)*(win+guard), 1)]];
    s_w = [[nan(win+guard, j);s_w],[s;nan((j+1)*(win+guard), 1)]];
  endfor

% re-adjust left side
  u_l = u_w((win+2*guard+2):end,:);
  s_l = s_w((win+2*guard+2):end,:);

% right side has same values as left side but shifted guard+1
% re-adjust right side
  u_r = [nan(win+guard, clusters);u_w(1:rows(u_l)-(win+guard),:)];
  s_r = [nan(win+guard, clusters);s_w(1:rows(s_l)-(win+guard),:)];


  Nu = sum(~isnan([u_l, u_r]), 2);

  p3 = find(Nu>2);
  p2 = find(Nu==2);

  u_f(p3) = maxk([u_l, u_r](p3,:), n_skip); % this operation puts data into rows instead of columns due to indexing
  u_f(p2) = mink([u_l, u_r](p2,:));

  s_f(p3) = maxk([s_l, s_r](p3,:), n_skip);
  s_f(p2) = nansum([s_l, s_r](p2, :), 2)/2;

% theta (threshold level)
  t = roundb(max([ones(1, numel(d)).*min_std;s_f/win^2]) .* th_factor + u_f/win)';
  mu = u_f';
  sigma = s_f';


  if( ~isempty(aux) && numel(aux)==numel(d) )
    aux_d = aux;
  else
    aux_d = zeros(numel(data), 1);
  endif

  file_info.header = { 'trace', 'data_o', 'threshold', 'blind_on', 'last' };
  file_info.data = [ d, t, aux_d ];

endfunction
% vim:tw=0:ts=2:sts=2:sw=2:et