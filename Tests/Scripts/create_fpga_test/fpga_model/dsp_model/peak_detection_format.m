function fmt = peak_detection_format()
% PEAK_DETECTION_FORMAT  Centralized format/precision/constant related to peak_detection

   fmt = struct;

   fmt.bin        = struct('integer', 10, 'fractional', 6 );
   fmt.mag        = struct('integer', 29, 'fractional', 0 );
   fmt.baseline   = struct('integer', 16, 'fractional', 0 ); % from blinder

% vim:tw=0:ts=3:sts=3:sw=3:et
