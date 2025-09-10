
import{Config} from "./config.js"
import{Editor} from "./editor.js"
import {Data} from './data.js'

let pointsGlobalConfig = new Config();
window.pointsGlobalConfig = pointsGlobalConfig;

pointsGlobalConfig.load();

document.documentElement.className="theme-"+pointsGlobalConfig.theme;

document.body.addEventListener('keydown', event => {
    if (event.ctrlKey && 'asdv'.indexOf(event.key) !== -1) {
      event.preventDefault()
    }
});

async function createMainEditor(){

  let template = document.querySelector('#editor-template');
  let maindiv  = document.querySelector("#main-editor");
  let main_ui = template.content.cloneNode(true);
  maindiv.appendChild(main_ui); // input parameter is changed after `append`

  let editorCfg = pointsGlobalConfig;

  let dataCfg = pointsGlobalConfig;
  
  let data = new Data(dataCfg);
  await data.init();

  let editor = new Editor(maindiv.lastElementChild, maindiv, editorCfg, data, "main-editor")
  window.editor = editor;
  editor.run();
  return editor;
} 



async function start(){
 console.log("🟢 START FUNCTION CALLED");

 let mainEditor = await createMainEditor();
 console.log("🔧 mainEditor created");

 const defaultFrame = "000000";

 try {

   await mainEditor.load_world("kitti", defaultFrame);
   console.log("✅ First load_world completed");


   setTimeout(() => {
     mainEditor.load_world("kitti", defaultFrame);
     console.log("🔁 Re-loaded frame to trigger image view");
   }, 250);

   setTimeout(() => {
     window.dispatchEvent(new Event('resize'));
     console.log("🔧 Resize event dispatched");
   }, 500);


   const url = new URL(window.location.href);
   if (!url.searchParams.has("scene")) {
     url.searchParams.set("scene", "kitti");
     url.searchParams.set("frame", defaultFrame);
     window.history.replaceState({}, '', url);
   }

 } catch (err) {
   console.error("❌ Error in auto-load sequence:", err);
 }
}

function autoPlay(editor) {
    if (isPlaying) return;
    isPlaying = true;
    currentFrame = startFrame;
    frameTimes = []; // 重置统计
    isMeasuring = false;

    async function playNextFrame() {
        if (currentFrame > endFrame) {
            console.log(`✅ Auto-play finished at frame ${endFrame.toString().padStart(6, '0')}`);

            printFPSStats();

            isPlaying = false;
            return;
        }

        const frameStr = currentFrame.toString().padStart(6, '0');
        console.log(`🎬 Playing frame: ${frameStr}`);

        // 🔍 是否进入统计阶段？
        const shouldMeasure = currentFrame >= startMeasureFrame;

        let startTime;
        if (shouldMeasure) {
            startTime = performance.now();
        }

        try {
            await editor.load_world("kitti", frameStr);

            if (editor.imageContextManager && editor.world) {
                editor.imageContextManager.attachWorld(editor.world);
                editor.imageContextManager.render_2d_image();
            }

            const url = new URL(window.location.href);
            url.searchParams.set("frame", frameStr);
            window.history.replaceState({}, '', url);

            window.dispatchEvent(new Event('resize'));

            if (shouldMeasure) {
                const endTime = performance.now();
                const processTime = endTime - startTime; // 实际耗时（ms）
                frameTimes.push(processTime);
                console.log(`📊 Frame ${frameStr}: process time = ${processTime.toFixed(2)}ms`);
            }

            currentFrame++;

            setTimeout(playNextFrame, interval);

        } catch (err) {
            console.error(`❌ Failed to load frame ${frameStr}:`, err);
            isPlaying = false;
        }
    }

    playNextFrame();
}

function printFPSStats() {
    if (frameTimes.length === 0) {
        console.log("📊 No frames measured.");
        return;
    }

    const firstMeasuredFrame = startMeasureFrame;
    const lastMeasuredFrame = endFrame;
    const totalFrames = frameTimes.length;
    const totalTimeMs = frameTimes.reduce((sum, t) => sum + t, 0);
    const avgProcessTimeMs = totalTimeMs / totalFrames;

    const avgFPS = 1000 / avgProcessTimeMs;

    console.log(`\n📈 FPS Statistics`);
    console.log(`------------------------------------------------`);
    console.log(`📊 Measurement Range: Frame ${firstMeasuredFrame} to Frame ${lastMeasuredFrame}`);
    console.log(`🔢 Total Frames Measured: ${totalFrames}`);
    console.log(`⏱️  Average Processing Time Per Frame: ${avgProcessTimeMs.toFixed(2)} ms`);
    console.log(`🎯 Average FPS: ${avgFPS.toFixed(2)}`);
    console.log(`------------------------------------------------`);
}

start();