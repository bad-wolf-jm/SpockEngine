function [d, sat_pulse, smpl_before, bl_before, bl_after, pulse_len, file_info] = blinder(data, cfg)
  % =============================================================================
  %  Copyright (c) LeddarTech Inc. All rights reserved.
  % =============================================================================
  %  PROPRIETARY DATA NOTICE
  %  No part of this file shall be used, communicated, reproduced
  %  or copied in any form or by any means without the prior written
  %  permission of LeddarTech Inc.
  % =============================================================================
  % Blinder
  %
  % [d, sat_pulse, bl_before, bl_after, file_info] = blinder(data, cfg)
  %
  % data : Trace data samples
  % cfg  : Configuration structure
  %
  % d        : Data post blinder
  % sat_pulse: Indicated which samples are actually clipped
  % bl_before: Mean of the win samples before saturation
  % bl_after : Mean of the win samples after bl_af_delay
  % file_info: File information for results writing
  %

  % convert to column vector
	sz = size(data);
	if sz(2) > sz(1)
	  d = data';
	else
	  d = data;
	endif

  sat_pulse   = zeros(numel(data), 1);
  bl_before   = zeros(numel(data), 1);
  bl_after    = zeros(numel(data), 1);
  blind_value = zeros(numel(data), 1);
  smpl_before = zeros(numel(data), 1);
  pulse_len   = zeros(numel(data), 1);

  file_info = {};

  if( cfg.block_enable )

    if( numel(data) <  sum([cfg.blind_cfg.clip_period, cfg.baseline_cfg.bl_af_delay, 2.^(cfg.baseline_cfg.win_size+1), cfg.baseline_cfg.guardlen_win_before, cfg.baseline_cfg.guardlen_win_after]) )
      error('Not enough data for blinder parameters')
    endif

    % find blind lengths
  	blind = find(diff(d>cfg.blind_cfg.th_blind_on)~=0) + 1; % diff moves everything left by one
  	if( mod(numel(blind), 2)>0 )
  		blind = [blind; numel(d)];	% signal blinded up until the end
  	endif;

    % Organize blinding indicators by columns
  	blind = reshape(blind, 2, []);

    % Merge consecutive small pulses (and fix LCA3 saturation plateau dent issue at the same time)
    blind_modif = 1;
    while(blind_modif == 1)
      blind_modif = 0;
      for i=1:columns(blind)-1
        if (blind(1,i+1)-blind(2,i) <= cfg.blind_cfg.clip_period+cfg.baseline_cfg.guardlen_win_after)
           blind(2,i) = blind(2,i+1);
           blind(:,i+1) = [];
           blind_modif = 1;
           break;
        endif
      endfor
    endwhile

    % figure out which ones require clipping
  	blinded = find(diff(blind)>=cfg.blind_cfg.clip_period);

  	% blinder's moving sum is filling before and emptying after
  	[ms_af, div_af, ms_bf, div_bf] = msum(d, 2.^cfg.baseline_cfg.win_size);

  	m_bf = ms_bf ./ div_bf;
  	m_af = ms_af ./ div_af;

    % a loop is more efficient here....
    for j=blinded

    	m_bf_pos = blind(1,j) - cfg.baseline_cfg.guardlen_win_before - 1;
    	m_af_pos = blind(1,j) + cfg.blind_cfg.clip_period + cfg.baseline_cfg.bl_af_delay;

    	m_bv_pos = blind(2,j) + cfg.baseline_cfg.guardlen_win_after;

      bl_value_period  = [ blind(1,j) + cfg.blind_cfg.clip_period : min( m_bv_pos - 1,numel(d)) ];
      side_info_period = [ blind(1,j):blind(1,j)+1 ];
      smpl_before_pos  = blind(1,j) - 1

    	if( m_bf_pos > 0 )
    		bl_before(side_info_period) = floor(mean(m_bf(m_bf_pos)));
    	endif

      if (cfg.baseline_cfg.bl_af_delay > 0) %Use baseline after computed at fixed time
    	   if( m_af_pos<numel(d) )
  	  	    bl_after(side_info_period) = floor(mean(m_af(m_af_pos)));
         else
            bl_after(side_info_period) = -(2^15);
    	   endif
      else
         if( m_bv_pos<numel(d) ) %Use blind value for baseline after
            bl_after(side_info_period) = floor(mean(m_af(m_bv_pos)));
         else
            bl_after(side_info_period) = -(2^15);
         endif
      endif

    	if( m_bv_pos<numel(d) )
  	  	  blind_value(bl_value_period) = floor(mean(m_af(m_bv_pos)));
      else
          blind_value(bl_value_period) = floor(mean(m_bf(m_bf_pos)));
      endif

    	d(bl_value_period) = blind_value(bl_value_period);
    	sat_pulse(side_info_period) = 1;
      pulse_len(side_info_period) = blind(2,j)-blind(1,j);
      smpl_before(side_info_period) = d(smpl_before_pos);
    endfor

  endif

  file_info.header = { 'trace'; 'data_o'; 'smpl_before'; 'bl_before'; 'bl_after'; 'pulse_len'; 'blind_on'; 'last' };
  file_info.data = [ d, smpl_before, bl_before, bl_after, pulse_len, sat_pulse ];

endfunction
% vim:tw=0:ts=2:sts=2:sw=2:et