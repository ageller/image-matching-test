/**
 * align_images.jsx
 *
 * Photoshop ExtendScript — Auto-Align Two Images with Opacity Overlay
 */

main();

function main() {

    // ── Step 1: Pick two files ────────────────────────────────────────────────

    alert("Select the REFERENCE image (the image other images will align TO).");
    var file1 = File.openDialog("Select the REFERENCE image");
    if (!file1) { alert("Cancelled."); return; }

    alert("Select the image TO BE ALIGNED (will be warped to match the reference).");
    var file2 = File.openDialog("Select the image to align");
    if (!file2) { alert("Cancelled."); return; }

    // ── Step 2: Open both files ───────────────────────────────────────────────

    var doc1 = app.open(file1);
    app.refresh();
    var doc2 = app.open(file2);
    app.refresh();

    // ── Step 3: Convert Background layers to normal layers ───────────────────

    app.activeDocument = doc1;
    if (doc1.activeLayer.isBackgroundLayer) {
        doc1.activeLayer.isBackgroundLayer = false;
    }
    app.refresh();

    app.activeDocument = doc2;
    if (doc2.activeLayer.isBackgroundLayer) {
        doc2.activeLayer.isBackgroundLayer = false;
    }
    app.refresh();

    // ── Step 4: Move doc2's layer into doc1 ──────────────────────────────────

    app.activeDocument = doc2;
    doc2.activeLayer.duplicate(doc1, ElementPlacement.PLACEATBEGINNING);
    app.refresh();
    doc2.close(SaveOptions.DONOTSAVECHANGES);
    app.refresh();

    // ── Step 5: Name layers ───────────────────────────────────────────────────
    // layers[0] = top    = duplicated from doc2 = "To Align"
    // layers[1] = bottom = original doc1 layer  = "Reference"

    app.activeDocument = doc1;
    app.refresh();
    doc1.layers[0].name = "To Align";
    doc1.layers[1].name = "Reference";
    app.refresh();

    // ── Step 6: Select both layers ───────────────────────────────────────────

    selectLayerByName("Reference", false);
    app.refresh();
    selectLayerByName("To Align", true);
    app.refresh();

    // ── Step 7: Auto-Align ───────────────────────────────────────────────────

    autoAlignLayers();
    app.refresh();

    // ── Step 8: Set "To Align" to 50% opacity ────────────────────────────────

    selectLayerByName("To Align", false);
    app.refresh();
    findLayerByName(doc1, "To Align").opacity = 75;
    app.refresh();

    // ── Step 9: Done ─────────────────────────────────────────────────────────

    alert(
        "Done!\n\n" +
        "The 'To Align' layer has been aligned to the Reference\n" +
        "and set to 75% opacity so both images are visible.\n\n" +
        "Adjust the opacity in the Layers panel as needed.\n" +
        "Use File > Save As to save the result."
    );
}


// ─── Helper: Select a layer by name via Action Descriptor ────────────────────

function selectLayerByName(layerName, addToSelection) {
    var desc = new ActionDescriptor();
    var ref  = new ActionReference();
    ref.putName(charIDToTypeID("Lyr "), layerName);
    desc.putReference(charIDToTypeID("null"), ref);
    if (addToSelection) {
        desc.putEnumerated(
            stringIDToTypeID("selectionModifier"),
            stringIDToTypeID("selectionModifierType"),
            stringIDToTypeID("addToSelection")
        );
    }
    desc.putBoolean(charIDToTypeID("MkVs"), false);
    executeAction(charIDToTypeID("slct"), desc, DialogModes.NO);
}


// ─── Helper: Auto-Align Layers — exact descriptor from ScriptListener ────────

function autoAlignLayers() {
    var desc = new ActionDescriptor();

    var ref = new ActionReference();
    ref.putEnumerated(
        charIDToTypeID("Lyr "),
        charIDToTypeID("Ordn"),
        charIDToTypeID("Trgt")
    );
    desc.putReference(charIDToTypeID("null"), ref);

    desc.putEnumerated(
        charIDToTypeID("Usng"),
        charIDToTypeID("ADSt"),
        stringIDToTypeID("ADSContent")
    );

    desc.putBoolean(stringIDToTypeID("alignToCanvas"), false);

    desc.putEnumerated(
        charIDToTypeID("Aply"),
        stringIDToTypeID("projection"),
        charIDToTypeID("Auto")
    );

    desc.putBoolean(stringIDToTypeID("vignette"),      false);
    desc.putBoolean(stringIDToTypeID("radialDistort"), false);

    executeAction(charIDToTypeID("Algn"), desc, DialogModes.NO);
}


// ─── Helper: Find a layer by name ────────────────────────────────────────────

function findLayerByName(doc, name) {
    for (var i = 0; i < doc.layers.length; i++) {
        if (doc.layers[i].name === name) {
            return doc.layers[i];
        }
    }
    return null;
}