
import metaworld
import mujoco

def list_cameras():
    env_name = 'pick-place-v3'
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name]()
    
    model = env.model
    n_cams = model.ncam
    
    print(f"Total cameras: {n_cams}")
    for i in range(n_cams):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"Camera {i}: {name}")

if __name__ == "__main__":
    list_cameras()
