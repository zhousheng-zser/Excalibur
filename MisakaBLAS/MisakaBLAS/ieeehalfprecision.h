#ifndef _ieeehalfprecision_h
#define _ieeehalfprecision_h

extern int singles2halfp(void *target, void *source, int numel);
extern int doubles2halfp(void *target, void *source, int numel);
extern int halfp2singles(void *target, void *source, int numel);
extern int halfp2doubles(void *target, void *source, int numel);

#endif
