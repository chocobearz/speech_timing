import logging
import multiprocessing
import gentle

def align(audiofile, txtfile, nthreads = multiprocessing.cpu_count(), conservative = False, disfluency = False, log = "INFO"): 
  """
    Align a transcript to audio by generating a new language model.

    Required arguments:
      Audio File: string
      Text File (has been modified to directly take a string): string
    Optional arguments:
      Number of alignment threads: int
      Conservative alignment: bool
      Include disfluencies (uh, um) in alignment: bool
      The log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL): string
"""
  
  logging.getLogger().setLevel(log)
  
  disfluencies = set(['uh', 'um'])
  
  def on_progress(p):
      for k,v in p.items():
          logging.debug("%s: %s" % (k, v))
  
  #in our use case we directly call the string
  #with open(args.txtfile, encoding="utf-8") as fh:
  #    transcript = fh.read()
  transcript = txtfile
  
  resources = gentle.Resources()
  logging.info("converting audio to 8K sampled wav")
  
  with gentle.resampled(audiofile) as wavfile:
      logging.info("starting alignment")
      aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=disfluency, conservative=conservative, disfluencies=disfluencies)
      result = aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)

  return(result)

