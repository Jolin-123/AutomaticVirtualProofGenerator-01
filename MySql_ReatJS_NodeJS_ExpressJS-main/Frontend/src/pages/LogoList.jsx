import React, { useEffect, useState} from 'react'
import axios from 'axios';
import './Student.css' ;
import { Link, useParams } from 'react-router-dom'
import Sidebar from '../components/Sidebar'


// pantone color and HEX object 
const colorsLib = {
    "PMS123": "#FDB927",
    "PMS268": "#552583",
    "Black": "#000000",
    "PMS5605C": "#203731",
    "PMS1235C": "#FFB612",
    "GOLD872C": "#ffd700",
    "PMS872C": "#ffd700",
    "PMS187C": "#EF3340",
    "PMS5535C": "#203731",
    

    "PMS202C": "#98002e",
    "PMS137C": "#f9a01b",
    "PMS123C": "#ffc72c",
    "PMSBlackC": "#000000",
    "PMS293C": "#006bb6",
    "PMS199C": "#ed174c",
    "PMS289C": "#002b5c",
    "PMS877C": "#c4ced4",
    "PMS268C": "#5a2b81",
    "PMS431C": "#63727a",
    "PMS282C": "#002d62",
    "PMSCoolGray5C": "#bec0c2",
    "PMS186C": "#c8102e",
    "PMS7687C": "#1d428a",
    "PMS356C": "#008348",
    "PMS874C": "#bb9753",
    "PMS174C": "#a73832",
    "PMS472C": "#fab383",
    "PMS2388C": "#0053bc",
    "yellow": "#ffc72c",
    "YELLOW": "#ffc72c",
    "PURPLE": "#552583",
    "white": "#FFFFFF",
    "WHITE": "#FFFFFF",
    "BLACK": "#000000"
  };



function LogoList() {

    const [data, setData] = useState([]);
  

//************** 1/27/2024 ************** ************** ************** **************
    // Working code
    // useEffect(() => {
    //     axios.get('http://localhost:8081/users')
    //     .then(res => {
    //       console.log("here is dataaaa:" + res.data); // Log the response data
    //       setData(res.data);     // Set the state with the response data
    //     })
    //     .catch(err => console.log(err));
    //   }, []);

    useEffect(() => {
        axios.get('http://localhost:3030/alllogo/')
            .then(res => {
                setData(res.data);
            })
            .catch(err => console.log(err));
    }, []);

    
      const handleDelete = async (id) => {
        try {
            await axios.delete(`http://localhost:3030/deleteLogoList/${id}`); //call the api
            window.location.reload();
        }catch(err){
            console.log("err from delete from logoList frontend -- ");
        }
      }

      function colorRec(color) {
        if (color != null) {
            // Create the rectangle element
            const colorBox = (
                <div className='colorBoxSmall' style={{ backgroundColor: colorsLib[color], color: 'white' }}>
                    {/* {color} */}
                </div>
            );
            return colorBox;
        } else {
            return null; // Return null if color is null
        }
    }
    


    return (
        <Sidebar> 
        <div className='container_main '> 
            
             <div className='table_container'>
             <h2 className='order_list_title'> Logo List </h2>

                <table className='list_table table_row'>  
                    <thead>
                    <tr>
                    <th>LogoId </th>
                    <th>Graphic </th>
                    <th>Description </th>
                    <th>Team </th>
                    <th>Width </th>
                    <th>Height </th>
                    <th>Image </th>

                    <th> C #1 </th>
                    <th> C #2  </th>
                    <th> C #3  </th>

                    <th> C #4  </th>
                    <th> C #5  </th>
                    <th> C #6  </th>
                    {/* maybe need to add one more  */}

                    <th> <Link to="/logoinput" className='btn btn-add'> Add+  </Link> </th>

                    </tr>
                    </thead>
                    
                    <tbody >
                    {
                        data.map((d, i) => (
                            <tr key={i} >
                                <td>{d.id}</td>
                                <td>{d.gname}</td>
                                <td>{d.logodesc}</td>
                                <td>{d.tname}</td>
                                <td>{d.width}</td>
                                <td>{d.height}</td>
                                {/* <td>{d.image}</td> */}
                                <td>
                                    {d.imagel && <img src={`http://localhost:3030/images/${d.imagel}`} alt="logo" style={{ width: "100px", height: "100px" }} />}
                                </td>

                                {/* <td>{d.c1} <div className='colorBox'> </div> </td> 
                                <td>{d.c2} <div className='colorBox'> </div> </td>
                                <td>{d.c3} <div className='colorBox'> </div> </td>
                                <td>{d.c4} <div className='colorBox'> </div> </td>
                                <td>{d.c5} <div className='colorBox'> </div> </td>
                                <td>{d.c6} <div className='colorBox'> </div> </td> */}
                            
                                <td> {d.c1} <div> { colorRec(d.c1) } </div> </td>
                                <td> {d.c2} <div> { colorRec(d.c2) } </div> </td>
                                <td> {d.c3} <div> { colorRec(d.c3) } </div> </td>
                                <td> {d.c4} <div> { colorRec(d.c4) } </div> </td>
                                <td> {d.c5} <div> { colorRec(d.c5) } </div> </td>
                                <td> {d.c6} <div> { colorRec(d.c6) } </div> </td> 
                            

                            <td className='button-container'>
                                {/* <button className='btn read'> Read </button> */}
                                
                                <Link to={`/updatelogo/${d.id}`}  className='btn update'> Update </Link>
                                <button className='btn delete' onClick={e=> handleDelete(d.id)}> Delete </button>
                            </td>
                        </tr>
                      ))
                      }

                    </tbody>
                 </table>
             
             </div>
        </div>
        </Sidebar>
    );
}

export default LogoList; 