from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    p = request.app.state.pipeline
    return {
        "status"      : "ok",
        "version"     : getattr(p, "version", "v4"),
        "classes"     : len(p.class_names),
        "onnx_shape"  : p.session.get_outputs()[0].shape,
    }


@router.get("/health/live")
async def liveness():
    """Kubernetes / Docker liveness probe — no pipeline check, just process alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(request: Request):
    """Readiness probe — only healthy once the pipeline is fully loaded."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        from fastapi import Response
        return Response(status_code=503, content="pipeline not ready")
    return {"status": "ready", "classes": len(pipeline.class_names)}
