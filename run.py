#!/usr/bin/env python
import logging
import os
import sys

LEVEL = logging.INFO
from allennlp.common.util import import_submodules

extra_libraries = ["lib"]
for library in extra_libraries:
    import_submodules(library)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position
from lib.commands.predict_with_vocab_expansion import PredictWithVocabExpansion

subcommand_overrides = {"predict-with-vocab-expansion": PredictWithVocabExpansion()}

if __name__ == "__main__":
    main(prog="allennlp",
         subcommand_overrides=subcommand_overrides)
