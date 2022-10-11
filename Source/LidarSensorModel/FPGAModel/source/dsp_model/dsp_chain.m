function [ file_info ] = dsp_chain(data, nfo, cfg, trace, first=0, gen_csv=1, gen_tile=1, vis=0)
% =============================================================================
%  Copyright (c) LeddarTech Inc. All rights reserved.
% =============================================================================
%  PROPRIETARY DATA NOTICE
%  No part of this file shall be used, communicated, reproduced
%  or copied in any form or by any means without the prior written
%  permission of LeddarTech Inc.
% =============================================================================
% FPGA DSP chain model
%
% [ file_info ] = dsp_chain(data, cfg)
%
% data: Incoming data
% nfo Per traces info
% cfg Configuration structure
% trace: Trace number
% gen_csv: Generate modules CSV files
% gen_tile: Generate tile output file
% vis Enable visualization when not than 0
%
% file_info : Result file otuput structure
%
  if( first )
    file_mode = 'w';
  else
    file_mode = 'a';
  endif

  % Stats
  [nfo.baseline, nfo.noise_std_dev, file_info.stats] = stats(data, cfg.statistics);

  % Static Noise
  [dsn, file_info.sn_rem] = static_noise(data, nfo.pd_number, cfg.sn_rem);

  % Blinder
  [db, sat_pulse, smpl_before,  bl_before, bl_after, pulse_len, file_info.blinder] = blinder(dsn, cfg.blinder);

  sideinfo.sat_pulse = sat_pulse;
  sideinfo.bl_before = bl_before;
  sideinfo.bl_after  = bl_after;
  sideinfo.pulse_len = pulse_len;
  sideinfo.smpl_before = smpl_before;

  % FIR
  [df, sat_pulse_d, file_info.fir] = fir(db, cfg.mfilter, sat_pulse);

  % CFAR
  [thld, ~,  ~, file_info.cfar] = cfar(df, cfg.cfar, sat_pulse_d);

  % Peak Detection
  [detections, nfo.leftovers, file_info.detect_max] = peak_detection(df, thld, sat_pulse_d, sideinfo, cfg.peak_detect);

  % Packets
  [packet, ~] = packetizer(cfg.peak_detect, nfo, detections, thld, df, db, [], data);


  if( gen_csv )
    for module=fieldnames(file_info)'
      if( isfield(file_info.(module{:}), 'header') )

        s = size(file_info.(module{:}).data);
        tinfo.trace = ones(s(1), 1).*trace;
        tinfo.last = [zeros(s(1)-1, 1); 1];

        for trace_info={'trace', 'last'}
          pos = find(strcmp(file_info.(module{:}).header, trace_info{:}));
          if( ~isempty(pos) )

            switch pos
              case 1
                file_info.(module{:}).data = [ tinfo.(trace_info{:}), file_info.(module{:}).data ];
              case s(1)
                file_info.(module{:}).data = [ file_info.(module{:}).data, tinfo.(trace_info{:}) ];
              otherwise
                if numel(file_info.(module{:}).data)
                  file_info.(module{:}).data = [ file_info.(module{:}).data(:, 1:pos-1), tinfo.(trace_info{:}), file_info.(module{:}).data(:, pos:end) ];
                endif
            endswitch

          endif
        endfor

        f = fopen(sprintf('result_%s.csv', module{:}), file_mode);
        if( first )
          fprintf(f, '%s\n', strjoin(file_info.(module{:}).header, ';'));
        endif
        if numel(file_info.(module{:}).data)
          dlmwrite(f, file_info.(module{:}).data, ';', '-append');
        endif
        fclose(f);
      endif
    endfor
  endif

  if( gen_tile )
    % Dump all traces into the same file to from a single tile
    f = fopen('tile.bin', [file_mode, 'b'], 'ieee-le');
    fwrite(f, packet, 'uint32', 0, 'ieee-le');
    fclose(f);
  endif

  if( vis>0 )
    figure(1);
    h = subplot(2, 1, 1); plot([data, db, df]);
    legend(h, 'Samples', 'Clipped', 'Filtered');
    h = subplot(2, 1, 2); plot([df, max(data)./2.*(thld<df), max(data)./2.*sat_pulse]);
    legend(h, 'Filtered', 'Above threshold', 'Saturated');
    title(sprintf('frame: %u, pd_number: %u, angpos_laser: %u, timestamp: %lu', nfo.frame_number, nfo.pd_number, nfo.angpos_laser, round(nfo.timestamp)));
    refresh;
  endif

endfunction
% vim:tw=0:ts=2:sts=2:sw=2:et
