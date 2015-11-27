#include "scip/scip.h"
#include "pricer_wrapper.h"


# include "c_library.h"

void c_pricer_redcost_wrapper_fn ( void * py_scip_redcost, cy_pricer_redcost_callback_t cy_callback )
{
   cy_callback ( py_scip_redcost, scipPricerRedcost(SCIP* _scip, SCIP_PRICER* _pricer, SCIP_Real* _lowerbound, SCIP_Bool* _stopearly, SCIP_RESULT* _result) ) ;
}
