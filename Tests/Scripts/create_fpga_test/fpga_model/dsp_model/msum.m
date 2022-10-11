function [ms_left, wl, ms_right, wr] = msum(data, win)
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% Movind sum with dividers for average
%
% [ms_left, wl, ms_right, wr] = msum(data, win)
%
% data        : Trace data samples
% win         : Number of maximum samples for moving sum
%
% ms_left     : Moving sum progressively decreasing at the end
% wl          : Left weights for average calculation
% ms_right    : Moving sum progressively increasing at the beginning
% wr          : Right weights for average calculation
%


%% moving sum with partial sums
% create array of sums of wvery power of 2 from win down to 1
	j = 1;
	for k=2.^[0:(log(win)./log(2))]
		ms{j} = conv(ones(k, 1), data);
		j = j + 1;
	endfor


% left window is decreasing; right window is increasing
% no way around this.....
	w = win;
% extract mid-section
	ms_left = ms{end}(win:end-(win-1));
	ms_right = ms_left;
% augment mid-sections with each start or end of partial sums above
	for j=numel(ms)-1:-1:1
		ms_left = [ms_left; ms{j}(end-(w-2):end-(w/2-1))];
		ms_right = [ms{j}(w/2:w-1); ms_right];
		w = w/2;
	endfor

% build weights window
% that is divisions for each sum to calculate mean value
	W = W = 2.^floor(log([1:win-1])./log(2));
	wl = [ones(1, numel(ms_left)-numel(W))*win, fliplr(W)]';
	wr = [W, ones(1, numel(ms_left)-numel(W))*win]';

endfunction