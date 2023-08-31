const fs = require('fs');
const sdk = require("microsoft-cognitiveservices-speech-sdk");
const directory = "/mnt/c/Users/ptut0/Documents/speech_timing/basline/data/txt/"

/*
  Arguments:
  template: if template use template mode
*/

let template = `<speak 
        xmlns="http://www.w3.org/2001/10/synthesis" 
        xmlns:mstts="http://www.w3.org/2001/mstts" 
        xmlns:emo="http://www.w3.org/2009/10/emotionml" 
        version="1.0" xml:lang="en-US"
        >
        <voice name="en-US-JennyNeural">
          <mstts:express-as style="Neutral" >
            TEXT
          </mstts:express-as>
        </voice>
        </speak>`

function synthesizeSpeech(ssml, audioFile) {
    const speechConfig = sdk.SpeechConfig.fromSubscription("bcded59e70d7427489c33925841a874d", "canadacentral");
    const audioConfig = sdk.AudioConfig.fromAudioFileOutput(audioFile);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
  
    synthesizer.speakSsmlAsync(
        ssml,
        result => {
            if (result.reason == sdk.ResultReason.SynthesizingAudioCompleted) {
                console.log("synthesis finished.");
                console.log(JSON.stringify(result));
            } else {
              console.error("Speech synthesis canceled, " + result.errorDetails);
            }
  
            synthesizer.close();
            synthesizer = null;
        },
        error => {
          console.trace("err - " + err);
          synthesizer.close();
          synthesizer = null;
        });
        console.log("Now synthesizing to: " + audioFile);
}

// fill in the script
(async ()=>{
  try {
    const files = await fs.promises.readdir(directory);
    for( const file of files ) {
      const script = await fs.promises.readFile(`${directory}${file}`, 'utf8');
      const ssml = template.replace(`TEXT`, `${script}`);
      console.log(ssml)
      const filename = file.replace(`.txt`, '')
      const audioDirectory = directory.replace(`txt`, `audio`)
      const out_file = `${audioDirectory}${filename}.wav`
      //generate audio
      synthesizeSpeech(ssml, out_file)
      }
  }
  catch( e ) {
    console.error( "We've thrown! Whoops!", e );
  }
})();
