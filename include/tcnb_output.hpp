#ifndef TCNB_OUTPUT_HPP
#define TCNB_OUTPUT_HPP

#include <cstdio>

inline void printheader(FILE *outfile, const char *string) {
  fprintf(outfile,
          "+--------------------------------------------------------------+\n");
  fprintf(outfile, "| %-60s |\n", string);
  fprintf(outfile,
          "+--------------------------------------------------------------+\n");
}

inline void printitem(FILE *outfile, const char *string) {
  fprintf(outfile, "  | %-49s", string);
}

inline void printpass(FILE *outfile, bool status) {
  if (status)
    fprintf(outfile, " [PASS] |\n");
  else
    fprintf(outfile, " [FAIL] |\n");
}

inline void printfooter(FILE *outfile) {
  fprintf(outfile,
          "  +----------------------------------------------------------+\n\n");
}

#endif
